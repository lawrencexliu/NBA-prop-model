import os
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

# -----------------------------
# CONFIG
# -----------------------------
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]  # remove "Playoffs" if you only want regular season

SAVE_CSV_BACKUP = True  # set False if you only want parquet
SLEEP_SECONDS = 1.0     # increase if you get rate limited

BASE_DIR = os.path.abspath(".")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"pull_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def safe_filename(season: str, season_type: str) -> str:
    return f"nba_player_gamelogs_{season}_{season_type.replace(' ', '_')}".lower()

def pull_one(season: str, season_type: str) -> pd.DataFrame:
    # Bulk endpoint (fast). Returns a single dataframe.
    resp = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = resp.get_data_frames()[0].copy()
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    return df

def main():
    frames = []

    for season in SEASONS:
        for stype in SEASON_TYPES:
            name = safe_filename(season, stype)
            out_parquet = os.path.join(RAW_DIR, f"{name}.parquet")
            out_csv = os.path.join(RAW_DIR, f"{name}.csv")

            log(f"\n--- Pulling: {season} | {stype} ---")

            try:
                df = pull_one(season, stype)

                # Save per-season/per-type
                df.to_parquet(out_parquet, index=False)
                log(f"Saved Parquet: {out_parquet} | rows={len(df):,}")

                if SAVE_CSV_BACKUP:
                    df.to_csv(out_csv, index=False)
                    log(f"Saved CSV:    {out_csv} | rows={len(df):,}")

                frames.append(df)

                time.sleep(SLEEP_SECONDS)

            except Exception as e:
                log(f"FAILED: {season} | {stype} | error={repr(e)}")
                # back off a bit on failures
                time.sleep(max(3.0, SLEEP_SECONDS * 3))

    if not frames:
        log("\nNo data pulled. Exiting.")
        return

    # Combine everything
    combined = pd.concat(frames, ignore_index=True)

    combined_parquet = os.path.join(RAW_DIR, "nba_player_gamelogs_2020-21_to_2025-26_all.parquet")
    combined.to_parquet(combined_parquet, index=False)
    log(f"\nSaved COMBINED Parquet: {combined_parquet} | rows={len(combined):,}")

    if SAVE_CSV_BACKUP:
        combined_csv = os.path.join(RAW_DIR, "nba_player_gamelogs_2020-21_to_2025-26_all.csv")
        combined.to_csv(combined_csv, index=False)
        log(f"Saved COMBINED CSV:    {combined_csv} | rows={len(combined):,}")

    # Quick sanity print
    log("\nColumns:")
    log(", ".join(combined.columns))

    log("\nDone ✅")

if __name__ == "__main__":
    main()
