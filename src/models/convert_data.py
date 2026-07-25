"""
One-off conversion of the trainable pickles to Parquet.

WHY THIS EXISTS
The pickles under data/trainable/ were written by a NumPy 2.x stack, so they can
only be read by NumPy >= 2. On macOS x86_64 the newest available PyTorch wheel is
2.2.2, which was compiled against the NumPy 1.x ABI and cannot interoperate with
NumPy 2 at all ("Failed to initialize NumPy: _ARRAY_API not found"). That leaves
no single NumPy version that can both open the data and run the models.

Parquet stores plain columnar data with no pickled NumPy objects, so a file
written under NumPy 2 reads back identically under NumPy 1.26. Running this once
in the NumPy 2 environment breaks the deadlock; afterwards the pipeline runs on
NumPy 1.26 + torch 2.2.2 and dataloaders.py picks up the Parquet files
automatically.

USAGE
  # in the environment that CAN read the pickles (NumPy 2.x)
  python src/models/convert_data.py

  # then downgrade and run as usual
  pip install "numpy<2"
  python src/models/train.py
"""

import os
import sys

import pandas as pd


DATA_DIR = "./data/trainable/"
FEATURE_PKL = "final_data_500.pkl"
EA_PKL = "earning_dates_500.pkl"
FEATURE_PARQUET = "final_data_500.parquet"
EA_PARQUET = "earning_dates_500.parquet"


def convert_features(src: str, dst: str) -> None:
    """dict[ticker -> DataFrame indexed by date]  ->  one long DataFrame."""
    data = pd.read_pickle(src)
    frames = []
    for ticker, df in data.items():
        out = df.copy()
        out.index.name = "date"
        out = out.reset_index()
        out.insert(0, "ticker", ticker)
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long.to_parquet(dst, index=False)
    print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}  "
          f"({len(data)} tickers, {len(long):,} rows, {len(long.columns)-2} features)")


def convert_ea(src: str, dst: str) -> None:
    """dict[ticker -> DataFrame with an 'Earnings Date' column]  ->  long DataFrame."""
    ea = pd.read_pickle(src)
    frames = []
    for ticker, df in ea.items():
        out = df.copy()
        # Drop the timezone WITHOUT converting: the original pipeline used the
        # local (US/Eastern) wall-clock time, so an after-close 20:00 announcement
        # belongs to that day, not the next one in UTC. Converting first would
        # shift 17 after-close announcements by a day.
        parsed = pd.to_datetime(out["Earnings Date"], errors="coerce")
        out["Earnings Date"] = parsed.apply(
            lambda x: x.tz_localize(None) if x is not pd.NaT and x.tzinfo else x
        )
        out = out[["Earnings Date"]].dropna()
        out.insert(0, "ticker", ticker)
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(dst, index=False)
    print(f"  {os.path.basename(src)} -> {os.path.basename(dst)}  "
          f"({len(ea)} tickers, {len(long):,} announcement dates)")


def main() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is required for Parquet.  pip install pyarrow")

    for name in (FEATURE_PKL, EA_PKL):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run this from the repository root.")

    print("Converting trainable data to Parquet...")
    convert_features(os.path.join(DATA_DIR, FEATURE_PKL), os.path.join(DATA_DIR, FEATURE_PARQUET))
    convert_ea(os.path.join(DATA_DIR, EA_PKL), os.path.join(DATA_DIR, EA_PARQUET))

    print("\nDone. The original pickles are untouched; dataloaders.py will now")
    print("prefer the Parquet files. You can downgrade NumPy:")
    print("    pip install 'numpy<2'")


if __name__ == "__main__":
    main()
