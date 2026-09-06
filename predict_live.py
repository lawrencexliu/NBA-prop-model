"""
predict_live.py
================
NBA Over/Under predictor that combines:
  1. Statistical distribution — fits actual game scores to normal distribution,
     computes exact probability of beating any entered line
  2. ML models (RF, LR, KNN) — weighted confidence based on historical patterns
  3. Combined final probability — weighted blend of statistical + ML signal
  4. Injury adjustment — reduces OVER probability based on injury status

Usage:
    python predict_live.py --player "LeBron James" --pts 25.5
    python predict_live.py --player "LeBron James" --pts 25.5 --reb 7.5 --ast 8.0
    python predict_live.py --player "LeBron James" --combo 41.5
    python predict_live.py --players "LeBron James" "Anthony Davis" --pts 22.5
"""

import argparse
import pickle
import warnings
import math
import numpy as np
import pandas as pd
from scipy import stats
from nba_live_fetcher import get_player_features, fetch_game_log, find_player_id

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Exact feature list from your trained models
# ─────────────────────────────────────────────
MODEL_FEATURE_COLS = [
    'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'STL', 'BLK',
    'BLKA', 'PF', 'PFD', 'PLUS_MINUS', 'NBA_FANTASY_PTS'
]

LIVE_TO_MODEL = {
    "win_rate_10g":            "WL",
    "MIN_avg_10g":             "MIN",
    "FGM_avg_10g":             "FGM",
    "FGA_avg_10g":             "FGA",
    "FG_PCT_avg_10g":          "FG_PCT",
    "FG3M_avg_10g":            "FG3M",
    "FG3A_avg_10g":            "FG3A",
    "FG3_PCT_avg_10g":         "FG3_PCT",
    "FTM_avg_10g":             "FTM",
    "FTA_avg_10g":             "FTA",
    "FT_PCT_avg_10g":          "FT_PCT",
    "OREB_avg_10g":            "OREB",
    "DREB_avg_10g":            "DREB",
    "TOV_avg_10g":             "TOV",
    "STL_avg_10g":             "STL",
    "BLK_avg_10g":             "BLK",
    "BLKA_avg_10g":            "BLKA",
    "PF_avg_10g":              "PF",
    "PFD_avg_10g":             "PFD",
    "PLUS_MINUS_avg_10g":      "PLUS_MINUS",
    "NBA_FANTASY_PTS_avg_10g": "NBA_FANTASY_PTS",
}


def build_feature_vector(live_features: dict, expected_features: list = None) -> pd.DataFrame:
    if expected_features is None:
        expected_features = MODEL_FEATURE_COLS
    row = {}
    for live_key, model_key in LIVE_TO_MODEL.items():
        if model_key in expected_features:
            row[model_key] = live_features.get(live_key, 0.0)
    for col in expected_features:
        if col not in row:
            row[col] = 0.0
    return pd.DataFrame([row])[expected_features]


# ─────────────────────────────────────────────
#  Statistical line probability
# ─────────────────────────────────────────────
def compute_line_probability(game_scores: list, line: float) -> dict:
    """
    Fits a normal distribution to the player's recent game scores
    and computes the exact probability of exceeding the entered line.

    Args:
        game_scores : list of recent actual scores (e.g. last 10 PTS values)
        line        : the prop line to beat (e.g. 25.5)

    Returns:
        dict with mean, std, prob_over, prob_under, z_score, confidence_interval
    """
    if len(game_scores) < 3:
        return {"error": "Not enough games for statistical analysis"}

    scores = np.array(game_scores, dtype=float)
    mean   = float(np.mean(scores))
    std    = float(np.std(scores, ddof=1))  # sample std deviation

    if std == 0:
        prob_over = 1.0 if mean > line else 0.0
    else:
        # P(X > line) using normal distribution survival function
        z_score   = (line - mean) / std
        prob_over = float(stats.norm.sf(z_score))  # 1 - CDF

    prob_under = 1.0 - prob_over
    z_score    = (line - mean) / std if std > 0 else 0.0

    # 80% confidence interval for next game score
    ci_low, ci_high = stats.norm.interval(0.80, loc=mean, scale=std) if std > 0 else (mean, mean)

    # Hit rate — how many of the last N games actually exceeded the line
    actual_hit_rate = float(np.mean(scores > line))

    return {
        "mean":             round(mean, 2),
        "std":              round(std, 2),
        "z_score":          round(z_score, 3),
        "prob_over":        round(prob_over, 4),
        "prob_under":       round(prob_under, 4),
        "ci_low":           round(ci_low, 1),
        "ci_high":          round(ci_high, 1),
        "actual_hit_rate":  round(actual_hit_rate, 3),
        "n_games":          len(scores),
    }


def get_raw_scores(player_name: str, stat: str = "PTS", n: int = 10) -> list:
    """
    Fetches raw per-game scores for a specific stat from the live game log.
    stat: 'PTS', 'REB', 'AST', or 'COMBO'
    """
    try:
        pid = find_player_id(player_name)
        df  = fetch_game_log(pid, n_games=n)
        if df.empty:
            return []
        if stat == "COMBO":
            scores = (df["PTS"] + df["REB"] + df["AST"]).dropna().tolist()
        else:
            scores = df[stat].dropna().tolist() if stat in df.columns else []
        return scores[:n]
    except Exception as e:
        print(f"[WARN] Could not fetch raw scores for {player_name}: {e}")
        return []


# ─────────────────────────────────────────────
#  ML model predictions
# ─────────────────────────────────────────────
def predict_with_model1(live_features: dict, model_path: str = "finalized_model_M1.sav") -> dict:
    try:
        model = pickle.load(open(model_path, "rb"))
    except FileNotFoundError:
        return {"error": f"Model file not found: {model_path}"}
    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else MODEL_FEATURE_COLS
    X = build_feature_vector(live_features, expected).fillna(0)
    prob = model.predict_proba(X)[0]
    return {
        "model":      "Random Forest (M1)",
        "prob_over":  round(float(prob[1]), 4),
        "prob_under": round(float(prob[0]), 4),
    }


def predict_with_model2(live_features: dict, model_path: str = "finalized_model_M2.sav") -> dict:
    try:
        bundle = pickle.load(open(model_path, "rb"))
    except FileNotFoundError:
        return {"error": f"Model file not found: {model_path}"}
    scaler, model, features = bundle["scaler"], bundle["model"], bundle["features"]
    X = build_feature_vector(live_features, features).fillna(0)
    prob = model.predict_proba(scaler.transform(X))[0]
    return {
        "model":      "Logistic Regression (M2)",
        "prob_over":  round(float(prob[1]), 4),
        "prob_under": round(float(prob[0]), 4),
    }


def predict_with_model3(live_features: dict, model_path: str = "finalized_model_M3.sav") -> dict:
    try:
        bundle = pickle.load(open(model_path, "rb"))
    except FileNotFoundError:
        return {"error": f"Model file not found: {model_path}"}
    scaler, model, features, k = bundle["scaler"], bundle["model"], bundle["features"], bundle.get("k", "?")
    X = build_feature_vector(live_features, features).fillna(0)
    prob = model.predict_proba(scaler.transform(X))[0]
    return {
        "model":      f"KNN K={k} (M3)",
        "prob_over":  round(float(prob[1]), 4),
        "prob_under": round(float(prob[0]), 4),
    }


# ─────────────────────────────────────────────
#  Combined statistical + ML probability
# ─────────────────────────────────────────────
def combine_probabilities(stat_prob: float, ml_probs: list,
                           stat_weight: float = 0.60) -> dict:
    """
    Combines statistical line probability with ML model probabilities.

    Statistical component (60% weight by default):
      - Directly answers "will he beat this specific line"
      - Based on actual game score distribution

    ML component (40% weight):
      - Answers "is this player in good form"
      - Based on shooting efficiency, usage, game profile

    Args:
        stat_prob   : probability from normal distribution (0-1)
        ml_probs    : list of ML model OVER probabilities
        stat_weight : weight given to statistical component (0-1)

    Returns:
        dict with combined probability and prediction
    """
    ml_weight  = 1.0 - stat_weight
    ml_avg     = float(np.mean(ml_probs)) if ml_probs else 0.5

    combined   = (stat_prob * stat_weight) + (ml_avg * ml_weight)
    combined   = round(combined, 4)

    prediction = "OVER" if combined > 0.5 else "UNDER"
    confidence = round(max(combined, 1 - combined), 4)

    return {
        "stat_prob":   round(stat_prob, 4),
        "ml_avg":      round(ml_avg, 4),
        "combined":    combined,
        "prediction":  prediction,
        "confidence":  confidence,
        "stat_weight": stat_weight,
        "ml_weight":   ml_weight,
    }


def apply_injury_adjustment(prob_over: float, live_features: dict) -> tuple:
    """Adjusts OVER probability down based on injury severity. Returns (adjusted_prob, note)."""
    severity = live_features.get("injury_severity", 0)
    flag     = live_features.get("injury_flag", 0)

    if not flag or severity == 0:
        return prob_over, None

    if severity == 3:
        return 0.0, "Player listed as OUT — forced UNDER"

    adjustment = {1: 0.05, 2: 0.12}.get(severity, 0)
    adjusted   = max(0.0, round(prob_over - adjustment, 4))
    note       = f"Injury adjusted (severity {severity}): -{adjustment*100:.0f}% OVER probability"
    return adjusted, note


# ─────────────────────────────────────────────
#  Full prediction pipeline
# ─────────────────────────────────────────────
def predict_player(
    player_name: str,
    model: str = "both",
    m1_path: str = "finalized_model_M1.sav",
    m2_path: str = "finalized_model_M2.sav",
    m3_path: str = "finalized_model_M3.sav",
    apply_injury: bool = True,
    pts_line=None, reb_line=None, ast_line=None, combo_line=None,
) -> dict:

    # Fetch live features
    live_features = get_player_features(player_name, verbose=True)
    if not live_features:
        return {"error": f"Could not fetch features for {player_name}"}

    results = {
        "player": player_name,
        "snapshot": {
            k: live_features.get(k, "N/A") for k in
            ["PTS_avg_10g", "REB_avg_10g", "AST_avg_10g",
             "COMBO_SCORE_avg_10g", "COMBO_SCORE_trend",
             "injury_flag", "injury_severity", "injury_description",
             "days_since_last_game"]
        },
        "prop_lines": {}
    }

    # ML model probabilities (performance profile signal)
    ml_probs = []
    ml_results = {}

    if model in ("m1", "both"):
        r1 = predict_with_model1(live_features, m1_path)
        if "error" not in r1:
            ml_probs.append(r1["prob_over"])
            ml_results["M1"] = r1

    if model in ("m2", "both"):
        r2 = predict_with_model2(live_features, m2_path)
        if "error" not in r2:
            ml_probs.append(r2["prob_over"])
            ml_results["M2"] = r2

    if model in ("m3", "both"):
        r3 = predict_with_model3(live_features, m3_path)
        if "error" not in r3:
            ml_probs.append(r3["prob_over"])
            ml_results["M3"] = r3

    results["ml_models"] = ml_results

    # Process each prop line
    stat_map = {
        "PTS":   pts_line,
        "REB":   reb_line,
        "AST":   ast_line,
        "COMBO": combo_line,
    }

    for stat, line in stat_map.items():
        if line is None:
            continue

        # Fetch raw game scores for this stat
        raw_scores = get_raw_scores(player_name, stat=stat, n=10)

        if not raw_scores:
            results["prop_lines"][stat] = {"error": "Could not fetch raw scores"}
            continue

        # Statistical probability of beating the line
        stat_analysis = compute_line_probability(raw_scores, line)

        if "error" in stat_analysis:
            results["prop_lines"][stat] = stat_analysis
            continue

        # Combine statistical + ML
        combined = combine_probabilities(
            stat_prob=stat_analysis["prob_over"],
            ml_probs=ml_probs,
            stat_weight=0.60
        )

        # Injury adjustment on final combined probability
        injury_note = None
        if apply_injury:
            adjusted_prob, injury_note = apply_injury_adjustment(
                combined["combined"], live_features
            )
            combined["combined"]   = adjusted_prob
            combined["prob_under"] = round(1 - adjusted_prob, 4)
            combined["prediction"] = "OVER" if adjusted_prob > 0.5 else "UNDER"
            combined["confidence"] = round(max(adjusted_prob, 1 - adjusted_prob), 4)

        results["prop_lines"][stat] = {
            "line":         line,
            "raw_scores":   raw_scores,
            "stat_analysis": stat_analysis,
            "combined":     combined,
            "injury_note":  injury_note,
        }

    return results


# ─────────────────────────────────────────────
#  Pretty print
# ─────────────────────────────────────────────
def print_report(results: dict):
    if "error" in results:
        print(f"\n  ✗ ERROR: {results['error']}\n")
        return

    print(f"\n{'═'*58}")
    print(f"  PREDICTION REPORT: {results.get('player', 'Unknown')}")
    print(f"{'═'*58}")

    snap = results.get("snapshot", {})
    print(f"\n  📊 Rolling Averages (last 10 games)")
    print(f"     PTS  : {snap.get('PTS_avg_10g', 'N/A')}")
    print(f"     REB  : {snap.get('REB_avg_10g', 'N/A')}")
    print(f"     AST  : {snap.get('AST_avg_10g', 'N/A')}")
    print(f"     COMBO: {snap.get('COMBO_SCORE_avg_10g', 'N/A')}")
    print(f"     Trend: {snap.get('COMBO_SCORE_trend', 'N/A')}  (5g vs 10g)")

    print(f"\n  🏥 Injury Status")
    inj  = snap.get("injury_flag", 0)
    sev  = snap.get("injury_severity", 0)
    desc = snap.get("injury_description", "healthy")
    print(f"     Status : {'⚠️  ON REPORT' if inj else '✅ Healthy'}")
    if inj:
        print(f"     Severity: {sev}/3  |  {desc}")

    # ML model signals
    ml = results.get("ml_models", {})
    if ml:
        print(f"\n  🤖 ML Model Signals (performance profile)")
        for key, r in ml.items():
            over  = r.get("prob_over", 0)
            bar   = "█" * int(over * 15)
            print(f"     {r['model']:<28} OVER {over*100:.1f}%  [{bar:<15}]")

    # Prop line results
    props = results.get("prop_lines", {})
    if props:
        print(f"\n  {'─'*54}")
        print(f"  🎯 PROP LINE PREDICTIONS")

        for stat, p in props.items():
            if "error" in p:
                print(f"\n  {stat}: {p['error']}")
                continue

            line    = p["line"]
            sa      = p["stat_analysis"]
            comb    = p["combined"]
            scores  = p["raw_scores"]

            pred    = comb["prediction"]
            conf    = comb["confidence"]
            c_over  = comb["combined"]
            c_under = round(1 - c_over, 4)

            bar_o = "█" * int(c_over  * 20)
            bar_u = "█" * int(c_under * 20)

            print(f"\n  {'─'*54}")
            print(f"  {stat}  —  Line: {line}")
            print(f"\n     Last {sa['n_games']} games: {[round(s,1) for s in scores]}")
            print(f"     Mean: {sa['mean']}  |  Std Dev: {sa['std']}")
            print(f"     80% range next game: {sa['ci_low']} – {sa['ci_high']}")
            print(f"     Hit rate vs {line}: {sa['actual_hit_rate']*100:.0f}% of last {sa['n_games']} games")

            print(f"\n     📐 Statistical probability : {sa['prob_over']*100:.1f}% OVER")
            print(f"     🤖 ML performance signal  : {comb['ml_avg']*100:.1f}% OVER")
            print(f"     ⚖️  Weight                 : {comb['stat_weight']*100:.0f}% stats / {comb['ml_weight']*100:.0f}% ML")

            print(f"\n     {'🔼 OVER' if pred == 'OVER' else '🔽 UNDER'}  —  {conf*100:.1f}% confident")
            print(f"     OVER  [{bar_o:<20}] {c_over*100:.1f}%")
            print(f"     UNDER [{bar_u:<20}] {c_under*100:.1f}%")

            if p.get("injury_note"):
                print(f"\n     ⚠️  {p['injury_note']}")

    print(f"\n{'═'*58}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Over/Under Predictor")
    parser.add_argument("--player",           type=str,   help="Single player name")
    parser.add_argument("--players",          type=str,   nargs="+", help="Multiple player names")
    parser.add_argument("--model",            type=str,   default="both", choices=["m1", "m2", "m3", "both"])
    parser.add_argument("--m1",               type=str,   default="finalized_model_M1.sav")
    parser.add_argument("--m2",               type=str,   default="finalized_model_M2.sav")
    parser.add_argument("--m3",               type=str,   default="finalized_model_M3.sav")
    parser.add_argument("--no-injury-adjust", action="store_true")
    parser.add_argument("--pts",              type=float, default=None, help="Points line e.g. 25.5")
    parser.add_argument("--reb",              type=float, default=None, help="Rebounds line e.g. 7.5")
    parser.add_argument("--ast",              type=float, default=None, help="Assists line e.g. 8.0")
    parser.add_argument("--combo",            type=float, default=None, help="PTS+REB+AST line e.g. 41.5")
    args = parser.parse_args()

    apply_inj = not args.no_injury_adjust
    targets   = [args.player] if args.player else (args.players or [])

    if not targets:
        print("Provide --player or --players.")
    else:
        for name in targets:
            res = predict_player(
                name,
                model=args.model,
                m1_path=args.m1,
                m2_path=args.m2,
                m3_path=args.m3,
                apply_injury=apply_inj,
                pts_line=args.pts,
                reb_line=args.reb,
                ast_line=args.ast,
                combo_line=args.combo,
            )
            print_report(res)
