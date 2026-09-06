"""
Lagged feature construction for NBA player prop prediction.

The original pipeline used same-game box-score columns (FGM, FGA, FG3M, FTM,
OREB, DREB, NBA_FANTASY_PTS, ...) as features while the target was derived from
that same game's PTS + REB + AST. Points are recoverable as 2*FGM + FG3M + FTM
and rebounds as OREB + DREB, so the model was reconstructing its own target
rather than forecasting it.

This module builds features from PRIOR games only. Every column is shifted so
that row i contains information available before game i tipped off.

Expect AUC in the 0.55-0.62 range. That is what an honest prop model looks like.
"""

import numpy as np
import pandas as pd

# Box-score columns to build rolling history from.
BOX_COLS = [
    'MIN', 'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
    'OREB', 'DREB', 'TOV', 'STL', 'BLK', 'PF', 'PLUS_MINUS',
    'COMBO',
]

WINDOWS = [3, 5, 10]


def prepare(df):
    """Sort chronologically and normalize the columns we depend on."""
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
    df = df.dropna(subset=['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    df['WL'] = df['WL'].map({'W': 1, 'L': 0}).astype(float)
    df['HOME'] = (~df['MATCHUP'].str.contains('@')).astype(int)
    df['OPPONENT'] = df['MATCHUP'].str.split().str[-1]
    df['COMBO'] = df['PTS'] + df['REB'] + df['AST']
    return df


def add_rolling_features(df):
    """
    Rolling means of prior games, per player.

    shift(1) before rolling is the critical line: without it the window
    includes the current game and the leak comes straight back.
    """
    g = df.groupby('PLAYER_ID')

    for col in BOX_COLS:
        prior = g[col].shift(1)
        for w in WINDOWS:
            df[f'{col}_avg{w}'] = (
                prior.groupby(df['PLAYER_ID'])
                     .rolling(w, min_periods=max(2, w // 2))
                     .mean()
                     .reset_index(level=0, drop=True)
            )

    # Volatility: a player averaging 25 with a std of 3 is a different bet
    # than one averaging 25 with a std of 12.
    prior_combo = g['COMBO'].shift(1)
    df['COMBO_std10'] = (
        prior_combo.groupby(df['PLAYER_ID'])
                   .rolling(10, min_periods=5)
                   .std()
                   .reset_index(level=0, drop=True)
    )

    # Short-window minus long-window: is the player trending up or down?
    df['COMBO_trend'] = df['COMBO_avg3'] - df['COMBO_avg10']
    df['MIN_trend'] = df['MIN_avg3'] - df['MIN_avg10']

    return df


def add_context_features(df):
    """Schedule and situation — all knowable before tipoff."""
    g = df.groupby('PLAYER_ID')

    df['DAYS_REST'] = g['GAME_DATE'].diff().dt.days.clip(upper=10)
    df['BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)
    df['GAMES_PLAYED'] = g.cumcount()

    # Rolling win rate over prior games (form proxy).
    prior_wl = g['WL'].shift(1)
    df['WIN_RATE_10'] = (
        prior_wl.groupby(df['PLAYER_ID'])
                .rolling(10, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
    )

    return df


def add_opponent_features(df):
    """
    Opponent strength, computed from games before the current date only.

    Expanding mean of COMBO allowed per opponent, shifted by one game so the
    current matchup never contributes to its own opponent rating.
    """
    opp = (
        df.sort_values('GAME_DATE')
          .groupby('OPPONENT')['COMBO']
          .apply(lambda s: s.shift(1).expanding(min_periods=20).mean())
          .reset_index(level=0, drop=True)
    )
    df['OPP_COMBO_ALLOWED'] = opp
    return df


def build_target(df, line_col='COMBO_avg10'):
    """
    Binary target: did the player exceed a line set before the game?

    Using COMBO_avg10 as a stand-in for a sportsbook line keeps the original
    framing. Swap in real closing lines when you have them — that is the only
    version that measures edge against a market rather than against an average.
    """
    df['TARGET'] = (df['COMBO'] > df[line_col]).astype(int)
    df['LINE'] = df[line_col]
    return df


def feature_columns(df):
    """Every column safe to feed the model. Nothing from the current game."""
    rolling = [c for c in df.columns if any(
        c.endswith(f'_avg{w}') for w in WINDOWS
    )]
    extras = [
        'COMBO_std10', 'COMBO_trend', 'MIN_trend',
        'DAYS_REST', 'BACK_TO_BACK', 'GAMES_PLAYED',
        'WIN_RATE_10', 'OPP_COMBO_ALLOWED', 'HOME', 'LINE',
    ]
    return rolling + [c for c in extras if c in df.columns]


def build(df):
    """Full pipeline: raw game logs in, model-ready frame out."""
    df = prepare(df)
    df = add_rolling_features(df)
    df = add_context_features(df)
    df = add_opponent_features(df)
    df = build_target(df)

    # Drop rows without enough history to fill the windows.
    feats = feature_columns(df)
    before = len(df)
    df = df.dropna(subset=feats + ['TARGET']).reset_index(drop=True)
    print(f'Dropped {before - len(df):,} rows lacking history '
          f'({len(df):,} remain)')

    return df, feats


def leakage_check(feats):
    """
    Fail loudly if a same-game column sneaks into the feature list.

    Run this before every training run. Leakage is easy to reintroduce and
    silent when it happens.
    """
    banned = set(BOX_COLS) | {'COMBO', 'WL', 'NBA_FANTASY_PTS', 'TARGET'}
    hits = [f for f in feats if f in banned]
    if hits:
        raise ValueError(f'Same-game columns in feature set: {hits}')
    print(f'Leakage check passed — {len(feats)} lagged features')


def chronological_split(df, test_frac=0.2):
    """
    Split by date, not at random.

    A random split lets the model train on March games and test on January
    ones, which is not a situation you ever face live. Time-series data gets
    a time-series split.
    """
    cutoff = df['GAME_DATE'].quantile(1 - test_frac)
    train = df[df['GAME_DATE'] <= cutoff]
    test = df[df['GAME_DATE'] > cutoff]
    print(f'Train: {len(train):,} rows through {cutoff.date()}')
    print(f'Test:  {len(test):,} rows after')
    return train, test


if __name__ == '__main__':
    from pathlib import Path

    DATA = Path(__file__).parent / 'data' / 'raw'
    files = sorted(DATA.glob('nba_player_gamelogs_*.csv'))
    if not files:
        raise FileNotFoundError(f'No game logs found in {DATA}')

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f'Loaded {len(raw):,} raw rows from {len(files)} file(s)')

    df, feats = build(raw)
    leakage_check(feats)
    train, test = chronological_split(df)

    df.to_csv(DATA.parent / 'processed' / 'features_lagged.csv', index=False)
