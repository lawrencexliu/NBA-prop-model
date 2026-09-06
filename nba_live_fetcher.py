"""
nba_live_fetcher.py
====================
Live data fetcher for the NBA Over/Under ML model.
Pulls real-time game logs + injury context from the NBA API (free, no key).

Usage:
    from nba_live_fetcher import get_player_features
    features = get_player_features("LeBron James")
"""

import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    leaguedashplayerstats,
    playervsplayer,
)
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import leagueinjuries

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
CURRENT_SEASON = "2024-25"
REQUEST_DELAY  = 0.7        # seconds between API calls (respect rate limits)

INJURY_SEVERITY_MAP = {
    "out":         3,
    "doubtful":    2,
    "questionable":1,
    "probable":    0,
    "day-to-day":  1,
    "gtd":         1,        # game-time decision
}

# ─────────────────────────────────────────────
#  Player Lookup
# ─────────────────────────────────────────────
def find_player_id(player_name: str) -> int:
    """Returns NBA player_id for a given full name. Raises if not found."""
    results = nba_players.find_players_by_full_name(player_name)
    if not results:
        raise ValueError(f"Player '{player_name}' not found in NBA API.")
    # Prefer active players
    active = [p for p in results if p["is_active"]]
    return (active[0] if active else results[0])["id"]


# ─────────────────────────────────────────────
#  Game Log Fetcher
# ─────────────────────────────────────────────
def fetch_game_log(player_id: int, season: str = CURRENT_SEASON, n_games: int = 20) -> pd.DataFrame:
    """
    Fetches the last n_games game log entries for a player.
    Returns a cleaned DataFrame sorted newest-first.
    """
    time.sleep(REQUEST_DELAY)
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        print(f"[WARN] Game log fetch failed (season {season}): {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Parse date and sort
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    # Numeric coercion
    num_cols = ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG_PCT",
                "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
                "OREB", "DREB", "TOV", "STL", "BLK", "PLUS_MINUS"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["WL"] = df["WL"].map({"W": 1, "L": 0})

    return df.head(n_games)


# ─────────────────────────────────────────────
#  Injury Fetcher
# ─────────────────────────────────────────────
def fetch_injury_report() -> pd.DataFrame:
    """
    Pulls the current NBA injury report.
    Returns a DataFrame with columns: PLAYER_NAME, TEAM, STATUS, REASON
    """
    time.sleep(REQUEST_DELAY)
    try:
        inj = leagueinjuries.LeagueInjuries()
        df = inj.get_data_frames()[0]
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Injury report fetch failed: {e}")
        return pd.DataFrame(columns=["PLAYER_NAME", "TEAM_ABBREVIATION", "RETURN_DATE", "DESCRIPTION"])


def get_player_injury_features(player_name: str, injury_df: pd.DataFrame) -> dict:
    """
    Returns injury-related features for a player given the current injury report.
    
    Returns dict with keys:
        injury_flag         : 1 if player appears on injury report, else 0
        injury_severity     : 0-3 scale (0=probable, 3=out)
        days_since_last_game: derived from game log (passed separately)
    """
    features = {"injury_flag": 0, "injury_severity": 0, "injury_description": "healthy"}

    if injury_df.empty:
        return features

    # Case-insensitive name match
    name_lower = player_name.lower()
    match = injury_df[injury_df["PLAYER_NAME"].str.lower().str.contains(name_lower, na=False)]

    if match.empty:
        return features

    row = match.iloc[0]
    features["injury_flag"] = 1

    desc = str(row.get("DESCRIPTION", "")).lower()
    features["injury_description"] = desc

    # Map description keywords to severity score
    severity = 0
    for keyword, score in sorted(INJURY_SEVERITY_MAP.items(), key=lambda x: -x[1]):
        if keyword in desc:
            severity = score
            break
    features["injury_severity"] = severity

    return features


# ─────────────────────────────────────────────
#  Rolling Feature Engineering
# ─────────────────────────────────────────────
def engineer_rolling_features(game_log: pd.DataFrame, windows: list = [5, 10]) -> dict:
    """
    Computes rolling averages and trend features from recent game log.
    Returns a flat dict of features ready for model input.
    """
    features = {}

    if game_log.empty or len(game_log) < 3:
        print("[WARN] Not enough games to compute rolling features.")
        return features

    stat_cols = ["PTS", "REB", "AST", "MIN", "FG_PCT", "FG3_PCT", "FT_PCT",
                 "TOV", "STL", "BLK", "PLUS_MINUS", "FGM", "FGA", "OREB", "DREB"]

    for window in windows:
        window_data = game_log.head(window)
        for col in stat_cols:
            if col in window_data.columns:
                val = window_data[col].mean()
                features[f"{col}_avg_{window}g"] = round(val, 4) if not np.isnan(val) else 0.0

    # Combo score (mirrors your model's TARGET logic)
    game_log["COMBO_SCORE"] = game_log["PTS"] + game_log["REB"] + game_log["AST"]
    features["COMBO_SCORE_avg_10g"] = round(game_log.head(10)["COMBO_SCORE"].mean(), 4)
    features["COMBO_SCORE_avg_5g"]  = round(game_log.head(5)["COMBO_SCORE"].mean(), 4)

    # Trend: last 5 avg vs last 10 avg (positive = hot streak)
    features["COMBO_SCORE_trend"] = round(
        features["COMBO_SCORE_avg_5g"] - features["COMBO_SCORE_avg_10g"], 4
    )

    # Days since last game (rest days)
    if "GAME_DATE" in game_log.columns and len(game_log) >= 1:
        last_game_date = game_log["GAME_DATE"].iloc[0]
        features["days_since_last_game"] = (datetime.today() - last_game_date).days
    else:
        features["days_since_last_game"] = 1

    # Games played this season
    features["games_played_season"] = len(game_log)

    # WL-based win rate
    if "WL" in game_log.columns:
        features["win_rate_10g"] = round(game_log.head(10)["WL"].mean(), 4)

    return features


# ─────────────────────────────────────────────
#  Main: Get All Features for a Player
# ─────────────────────────────────────────────
def get_player_features(
    player_name: str,
    season: str = CURRENT_SEASON,
    injury_df: pd.DataFrame = None,
    verbose: bool = True,
) -> dict:
    """
    Master function. Returns a complete feature dict for a player,
    combining live game log stats + injury context.

    Args:
        player_name  : Full player name (e.g. "LeBron James")
        season       : NBA season string (e.g. "2024-25")
        injury_df    : Pre-fetched injury report DataFrame (optional, avoids extra API call)
        verbose      : Print status messages

    Returns:
        dict of features ready to pass to your ML models
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"  Fetching features for: {player_name}")
        print(f"{'='*50}")

    # 1. Player ID
    try:
        pid = find_player_id(player_name)
        if verbose: print(f"  ✓ Player ID: {pid}")
    except ValueError as e:
        print(f"  ✗ {e}")
        return {}

    # 2. Game log
    game_log = fetch_game_log(pid, season=season)
    if game_log.empty:
        print(f"  ✗ No game log data found for {player_name} in {season}")
        return {}
    if verbose: print(f"  ✓ Game log: {len(game_log)} games fetched")

    # 3. Injury report
    if injury_df is None:
        if verbose: print(f"  ↻ Fetching injury report...")
        injury_df = fetch_injury_report()
        if verbose: print(f"  ✓ Injury report: {len(injury_df)} entries")

    # 4. Feature engineering
    rolling_features = engineer_rolling_features(game_log)
    injury_features  = get_player_injury_features(player_name, injury_df)

    # 5. Merge everything
    all_features = {
        "player_name": player_name,
        "player_id":   pid,
        "season":      season,
        "as_of_date":  datetime.today().strftime("%Y-%m-%d"),
        **rolling_features,
        **injury_features,
    }

    if verbose:
        print(f"\n  📊 Rolling Stats (last 10g avg):")
        for k in ["PTS_avg_10g", "REB_avg_10g", "AST_avg_10g", "MIN_avg_10g"]:
            if k in all_features:
                print(f"     {k:<20}: {all_features[k]}")
        print(f"\n  🏥 Injury Status:")
        print(f"     injury_flag       : {all_features.get('injury_flag', 0)}")
        print(f"     injury_severity   : {all_features.get('injury_severity', 0)} / 3")
        print(f"     description       : {all_features.get('injury_description', 'healthy')}")
        print(f"     days_since_game   : {all_features.get('days_since_last_game', '?')}")
        print(f"     COMBO trend       : {all_features.get('COMBO_SCORE_trend', '?')}")

    return all_features


# ─────────────────────────────────────────────
#  Batch: Get Features for Multiple Players
# ─────────────────────────────────────────────
def get_batch_features(player_names: list, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Fetch features for a list of players. Shares one injury report fetch.
    Returns a DataFrame (one row per player).
    """
    print(f"Fetching shared injury report...")
    injury_df = fetch_injury_report()
    print(f"  ✓ {len(injury_df)} players on injury report\n")

    rows = []
    for name in player_names:
        features = get_player_features(name, season=season, injury_df=injury_df, verbose=True)
        if features:
            rows.append(features)
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(rows)
