import os
import re
import pickle
import pandas as pd
from collections import defaultdict

DATA_DIR = "/home/azureuser/cloudfiles/code/Users/manuel.noseda/DL_project_news"
OUTPUT_FILE = os.path.join(DATA_DIR, "WatchListNews_FULL_PERIOD.pkl")

# Format example: WatchListNews_20241001_20241211_100.pkl
FILENAME_RE = re.compile(r"WatchListNews_(\d{8})_(\d{8})_\d+\.pkl")



# SCAN DIR + GROUP FILES BY PERIOD

period_files = defaultdict(list)
all_start_dates, all_end_dates = [], []

for fname in os.listdir(DATA_DIR):
    m = FILENAME_RE.match(fname)
    if not m:
        continue

    start, end = m.groups()
    period_files[(start, end)].append(os.path.join(DATA_DIR, fname))
    all_start_dates.append(start)
    all_end_dates.append(end)

if not period_files:
    raise RuntimeError("No valid files found")

global_start = min(all_start_dates)
global_end = max(all_end_dates)
global_start_ts = pd.to_datetime(global_start)
global_end_ts = pd.to_datetime(global_end)


# BUILD ONE FINAL DICT: TICKER → FULL DF

final_dict = {}

for period, files in sorted(period_files.items()):
    # merge batches inside this period
    period_dict = {}

    for path in files:
        with open(path, "rb") as f:
            batch = pickle.load(f)  

        for ticker, df in batch.items():
            if ticker in period_dict:
                period_dict[ticker] = pd.concat(
                    [period_dict[ticker], df], ignore_index=True
                )
            else:
                period_dict[ticker] = df.copy()

    # merge this period into global dict
    for ticker, df in period_dict.items():
        if ticker in final_dict:
            final_dict[ticker] = pd.concat(
                [final_dict[ticker], df], ignore_index=True
            )
        else:
            final_dict[ticker] = df.copy()


# CLEAN EACH TICKER DF

clean_final_dict = {}

for ticker, df in final_dict.items():

    # skip invalid or empty dfs
    if "event_time" not in df.columns or "id" not in df.columns:
        continue

    df["event_time"] = (
        pd.to_datetime(df["event_time"], errors="coerce", utc=True)
          .dt.tz_convert(None)
    )

    mask = df["event_time"].between(
        global_start_ts, global_end_ts, inclusive="both"
    )

    df = (
        df[mask]
        .drop_duplicates(subset="id", keep="first")
        .sort_values("event_time", ascending=False)
        .reset_index(drop=True)
    )

    if not df.empty:
        clean_final_dict[ticker] = df

final_dict = clean_final_dict


# SAVE

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(final_dict, f)

print(f"DONE — {len(final_dict)} tickers, saved to {OUTPUT_FILE}")


