
Points are recoverable as `2*FGM + FG3M + FTM`. Rebounds are exactly `OREB + DREB`. `NBA_FANTASY_PTS` is a weighted sum containing points, rebounds, and assists directly. Two of the three target components were arithmetically derivable from the features, and the third leaked in through fantasy points.

The model was not forecasting. It was reconstructing a number it had already been handed, then comparing it to a threshold.

### The signal that gave it away

Alongside the ML models, the project used a statistical baseline that fit a normal distribution to the player's prior scoring and computed the probability of clearing the line. That baseline scored **0.523 AUC** — a coin flip.

Two methods on the same task, one at 0.84 and one at 0.52, is not a story about model quality. The baseline only had access to prior games. The ML models had access to the current one. That gap *was* the leakage, visible the whole time.

A second tell: all four v1 models landed within 0.02 AUC of each other. Model families that different should disagree more. They agreed because they were all doing the same arithmetic.

### v2: rebuilt

Every feature now comes from prior games only. The core of it, in `build_features.py`:

```python
prior = grouped[col].shift(1)          # push each player's stats down one row
rolling = prior.rolling(window).mean() # window now ends at game i-1
```

Without `shift(1)`, the rolling window includes the current game and the leak returns immediately.

The train/test split also changed from random to chronological. A random split trains on March games and tests on January ones — a situation you never face live. With per-player rolling features it's worse than it sounds, because the same player's adjacent games land on both sides of the split and share overlapping windows.

A `leakage_check()` guard now runs before every training run and raises if any same-game column reaches the feature list:

```python
def leakage_check(feats):
    banned = set(BOX_COLS) | {'COMBO', 'WL', 'NBA_FANTASY_PTS', 'TARGET'}
    hits = [f for f in feats if f in banned]
    if hits:
        raise ValueError(f'Same-game columns in feature set: {hits}')
```

### What the difference means

| | v1 | v2 |
|---|---|---|
| Features | Same-game box score | Prior games only |
| Split | Random, `random_state=42` | Chronological by date |
| Best AUC | 0.8425 | 0.7348 |
| Spread across models | 0.018 | 0.099 |

The widened spread is the useful diagnostic. When features carry real signal, different model families find different amounts of it. When they're all reconstructing an identity, they converge.

---

## What 0.7348 actually measures

Worth stating plainly, because the number is higher than a genuine prop model should reach.

The target is "did the player beat their trailing 10-game average." A trailing average is a **stale benchmark**. When a player's role expands — returning from injury, a teammate goes down, a rotation change — they beat that average repeatedly until it catches up. The `COMBO_trend` feature (3-game average minus 10-game average) captures exactly this.

So the model is substantially detecting that a lagging indicator lags. That is real and learnable, and it is not the same as beating a sportsbook.

A real prop line already prices in the injury, the role change, and the trend — the book has the same information and more. Evaluated against actual closing lines rather than a rolling average, this number would fall much closer to 0.5. **That comparison hasn't been run, because the historical line data isn't in this project yet.** Until it is, treat 0.7348 as a benchmark against a rolling average, not as evidence of market edge.

---

## Features

73 features, all lagged. Built by `build_features.py`:

**Rolling box-score averages** — 3, 5, and 10-game means for 21 columns (minutes, shooting splits, rebounds, turnovers, steals, blocks, fouls, plus/minus, fantasy points, and the PTS+REB+AST combo).

**Form and volatility** — `COMBO_std10` (consistency: averaging 25 with a std of 3 is a different proposition than averaging 25 with a std of 12), `COMBO_trend` and `MIN_trend` (short window minus long window).

**Schedule context** — days rest, back-to-back flag, games played, rolling win rate.

**Matchup** — opponent's expanding average COMBO allowed, shifted so tonight's game never contributes to the rating used to predict it. Home/away flag.

**The line itself** — the threshold the model is being asked about.

---

## Setup

```bash
git clone https://github.com/lawrencexliu/NBA-prop-model.git
cd NBA-prop-model

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running it

```bash
# 1. Pull game logs from the NBA Stats API
python pull_nba_gamelogs.py

# 2. Build lagged features (also runs standalone for inspection)
python build_features.py

# 3. Train and evaluate all three models
jupyter notebook nba_ml_models_combined.ipynb

# 4. Predict against a line
python predict_live.py --player "LeBron James" --pts 25.5
```

Steps 1 and 3 are required before step 4 — `predict_live.py` loads the pickled models and reports an error per model if they're missing rather than failing outright.

**The dataset is not committed.** It's roughly 332,000 rows across 14 seasons of game logs. Step 1 regenerates it.

---

## Repo contents

| File | Purpose |
|---|---|
| `build_features.py` | Lagged feature pipeline, leakage guard, chronological split |
| `pull_nba_gamelogs.py` | Pulls player game logs via `nba_api` |
| `nba_ml_models_combined.ipynb` | Training, evaluation, calibration, model export |
| `model_analysis.ipynb` | ROC, precision-recall, confusion matrices, calibration curves |
| `nba_live_fetcher.py` | Live last-10-game averages and ESPN injury report |
| `predict_live.py` | Takes a player and a line, returns a probability |

---

## Known limitations

- **Not validated against real lines.** The benchmark is a rolling average, not a sportsbook number. This is the single biggest gap and the next thing worth building.
- **Calibration is unverified.** AUC says the model ranks OVERs above UNDERs; it says nothing about whether "62% confident" means 62% of the time. For sizing a position, calibration matters more than ranking.
- **Rolling windows cross season boundaries.** Features are grouped by `PLAYER_ID` across all 14 seasons, so a player's last 2019 game feeds the average for his first 2020 game. Grouping by `(PLAYER_ID, SEASON)` would be cleaner at the cost of more dropped rows.
- **No lineup or usage context.** Nothing about who else is playing, which is often the largest driver of a single game's box score.
- **14 seasons of data, one model.** Pace and rule environments differ substantially across that span; the model doesn't account for era.

---

## Built with

Python · pandas · NumPy · scikit-learn · matplotlib · nba_api
MARKDOWN_EOF

wc -l README.md