"""
Build an earnings-event table (features + label) from:

1) final_data_500.pkl:
   dict[ticker] -> DataFrame indexed by trading-day DatetimeIndex, with columns like
   positive/negative/neutral + fundamentals + adj_price + MA3/MA6.

2) earning_dates_500.pkl:
   dict[ticker] -> DataFrame with column "Earnings Date" (tz-aware, America/New_York).

Output:
  A single event-level DataFrame with one row per earnings event in the modeling window:
    - ticker, earnings_ts_ny, earnings_day (mapped trading day), event_type (BMO/AMC), feature_day
    - return, label_up (binary), abs_return
    - aggregated sentiment over a lookback window (mean/std/trend + pos-neg)
    - snapshot of per-day features from final_data at feature_day (prefixed "f_")
    - event_id (ticker + timestamp) for debugging

Leakage controls (defaults are conservative):
  - For BMO: exclude T-1 (day before earnings), so last included sentiment day is T-2
  - For AMC: exclude T (earnings day), so last included sentiment day is T-1
  - Features used:
        BMO -> previous trading day (pre-announcement close)
        AMC -> same trading day (close before announcement)
"""

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -------------------------- config --------------------------

FINAL_DATA = Path("./data/trainable/final_data_500.pkl")
EARNING_DATES = Path("./data/trainable/earning_dates_500.pkl")
OUT = Path("./data/trainable/event_table_500.parquet")

LOOKBACK_DAYS = 30
PRICE_COL = "adj_price"

KEEP_UNKNOWN_TIMING = False
NY_TZ = "America/New_York"


# -------------------------- helpers --------------------------

def load_pickle(path: Path) -> Any:
    """Load a pickle file from the given path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a tz-naive DatetimeIndex sorted in ascending order."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    if df.index.tz is not None:
        # Keep trading index tz-naive for consistent comparisons
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df.sort_index()


def to_ny_timestamp(x: Any) -> pd.Timestamp:
    """Convert input to a pandas Timestamp in NY timezone. If input is NaN, returns NaT."""
    if pd.isna(x):
        return pd.NaT
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def classify_event_type(earn_ts_ny: pd.Timestamp) -> str:
    """Infer BMO vs AMC from NY timestamp. BMO: <09:30, AMC: >=16:00, else UNKNOWN."""
    mins = 60 * int(earn_ts_ny.hour) + int(earn_ts_ny.minute)
    if mins < 9 * 60 + 30:
        return "BMO"
    if mins >= 16 * 60:
        return "AMC"
    return "UNKNOWN"


def map_to_next_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
    """Map a (tz-naive) midnight date to the next available trading day in trading_index."""
    day = pd.Timestamp(day.date())
    pos = int(trading_index.searchsorted(day, side="left"))
    if pos < 0:
        pos = 0
    if pos >= len(trading_index):
        pos = len(trading_index) - 1
    return trading_index[pos]


def previous_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """Return the previous trading day before 'day', or None if none exists."""
    pos = int(trading_index.searchsorted(day, side="left"))
    if pos <= 0:
        return None
    if pos < len(trading_index) and trading_index[pos] == day:
        return trading_index[pos - 1]
    return trading_index[pos - 1]


def next_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """Return the next trading day after 'day', or None if none exists."""
    pos = int(trading_index.searchsorted(day, side="left"))
    if pos >= len(trading_index):
        return None
    if trading_index[pos] == day:
        if pos + 1 >= len(trading_index):
            return None
        return trading_index[pos + 1]
    return trading_index[pos]


def linreg_slope(y: np.ndarray) -> float:
    """Slope of y on x where x = 0..n-1. Returns NaN if ill-posed."""
    n = len(y)
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    y = y.astype(float)
    vx = x - x.mean()
    denom = float((vx * vx).sum())
    if denom <= 0.0:
        return float("nan")
    return float((vx * (y - y.mean())).sum() / denom)


@dataclass(frozen=True)
class BuildConfig:
    lookback_days: int = 30
    price_col: str = "adj_price"
    sentiment_cols: tuple[str, str, str] = ("positive", "negative", "neutral")


def sentiment_aggregates(df: pd.DataFrame, last_included_day: pd.Timestamp, cfg: BuildConfig) -> dict[str, float]:
    """Aggregate sentiment over a fixed lookback window ending at last_included_day (inclusive)."""
    pos_col, neg_col, neu_col = cfg.sentiment_cols
    for c in (pos_col, neg_col, neu_col):
        if c not in df.columns:
            raise KeyError(f"Missing sentiment column: {c}")

    end_day = pd.Timestamp(last_included_day)
    start_day = end_day - pd.Timedelta(days=cfg.lookback_days - 1)

    window = df.loc[(df.index >= start_day) & (df.index <= end_day), [pos_col, neg_col, neu_col]]
    out: dict[str, float] = {}

    if len(window) == 0:
        for name in ["mean", "std", "trend"]:
            out[f"sent_pos_{name}"] = float("nan")
            out[f"sent_neg_{name}"] = float("nan")
            out[f"sent_neu_{name}"] = float("nan")
        out["sent_pos_minus_neg_mean"] = float("nan")
        out["sent_days_used"] = 0.0
        return out

    out["sent_days_used"] = float(len(window))
    for col, pref in [(pos_col, "pos"), (neg_col, "neg"), (neu_col, "neu")]:
        s = window[col].astype(float)
        out[f"sent_{pref}_mean"] = float(s.mean())
        out[f"sent_{pref}_std"] = float(s.std(ddof=0))
        out[f"sent_{pref}_trend"] = linreg_slope(s.to_numpy())

    out["sent_pos_minus_neg_mean"] = out["sent_pos_mean"] - out["sent_neg_mean"]
    return out


def compute_event_return_and_label(
    df: pd.DataFrame, event_day: pd.Timestamp, event_type: str, cfg: BuildConfig
) -> tuple[float, int, float, pd.Timestamp] | None:
    """
    Returns (ret, label_up, abs_ret, feature_day), or None if not computable.

    feature_day:
      - BMO: previous trading day (pre-announcement close)
      - AMC: same trading day (close before announcement)
    """
    if cfg.price_col not in df.columns:
        raise KeyError(f"Missing price column '{cfg.price_col}'")

    idx = df.index
    if event_day not in idx:
        return None

    if event_type == "BMO":
        prev_day = previous_trading_day(idx, event_day)
        if prev_day is None:
            return None
        p0 = float(df.at[prev_day, cfg.price_col])
        p1 = float(df.at[event_day, cfg.price_col])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
            return None
        ret = p1 / p0 - 1.0
        return (float(ret), int(ret > 0.0), float(abs(ret)), prev_day)

    if event_type == "AMC":
        nxt_day = next_trading_day(idx, event_day)
        if nxt_day is None:
            return None
        p0 = float(df.at[event_day, cfg.price_col])
        p1 = float(df.at[nxt_day, cfg.price_col])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
            return None
        ret = p1 / p0 - 1.0
        return (float(ret), int(ret > 0.0), float(abs(ret)), event_day)

    return None


def event_id(ticker: str, ts_ny: pd.Timestamp) -> str:
    """Create a unique event ID from ticker and NY timestamp."""
    return f"{ticker}_{ts_ny.strftime('%Y%m%dT%H%M%S%z')}"


# -------------------------- main build --------------------------

def build_event_table(
    final_data: dict[str, pd.DataFrame],
    earning_dates: dict[str, pd.DataFrame],
    cfg: BuildConfig,
    drop_unknown_timing: bool = True,
) -> pd.DataFrame:
    """Build the event table DataFrame from final_data and earning_dates."""
    tickers = sorted(set(final_data.keys()) & set(earning_dates.keys()))
    rows: list[dict[str, object]] = []

    dropped_nat = 0
    dropped_unknown = 0
    dropped_outside = 0
    dropped_return = 0

    for t in tickers:
        df = ensure_datetime_index(final_data[t])

        start_day = df.index.min()
        end_day = df.index.max()

        e = earning_dates[t].copy()
        if "Earnings Date" not in e.columns:
            raise KeyError(f"{t}: earnings df missing 'Earnings Date' column")

        earn_ts_ny = e["Earnings Date"].map(to_ny_timestamp)

        for ts_ny in earn_ts_ny:
            if pd.isna(ts_ny):
                dropped_nat += 1
                continue

            event_type = classify_event_type(ts_ny)
            if event_type == "UNKNOWN":
                if drop_unknown_timing:
                    dropped_unknown += 1
                    continue

            calendar_day = pd.Timestamp(ts_ny.date())  # naive midnight NY calendar day

            # Filter earnings to our coverage window up-front (avoid mapping 2021 -> index[0]).
            if calendar_day < start_day or calendar_day > end_day:
                dropped_outside += 1
                continue

            # Map to next trading day in our index (handles weekends/holidays).
            event_day = map_to_next_trading_day(df.index, calendar_day)

            ret_info = compute_event_return_and_label(df, event_day, event_type, cfg)
            if ret_info is None:
                dropped_return += 1
                continue
            ret, label_up, abs_ret, feature_day = ret_info

            # Explicit leakage rule for sentiment last included day:
            # - BMO: exclude T-1 => last included = T-2
            # - AMC: exclude T   => last included = T-1
            if event_type == "BMO":
                last_included = event_day - pd.Timedelta(days=2)
            else:  # AMC or UNKNOWN (if kept)
                last_included = event_day - pd.Timedelta(days=1)

            sent_feats = sentiment_aggregates(df, last_included, cfg)

            snap = df.loc[feature_day].to_dict()
            snap = {f"f_{k}": v for k, v in snap.items()}

            row: dict[str, object] = {
                "event_id": event_id(t, ts_ny),
                "ticker": t,
                "earnings_ts_ny": ts_ny,
                "event_type": event_type,
                "calendar_day_ny": calendar_day,
                "earnings_day": event_day,
                "feature_day": feature_day,
                "return": float(ret),
                "abs_return": float(abs_ret),
                "label_up": int(label_up),
            }
            row.update(sent_feats)
            row.update(snap)
            rows.append(row)

    out = pd.DataFrame(rows)
    print(
        f"[INFO] dropped: NaT={dropped_nat}, UNKNOWN={dropped_unknown}, outside_window={dropped_outside}, "
        f"no_return={dropped_return}"
    )

    if len(out) == 0:
        return out

    out = out.sort_values(["earnings_day", "ticker"]).reset_index(drop=True)

    # Cast timestamps
    out["earnings_ts_ny"] = pd.to_datetime(out["earnings_ts_ny"])
    out["earnings_day"] = pd.to_datetime(out["earnings_day"])
    out["feature_day"] = pd.to_datetime(out["feature_day"])
    out["calendar_day_ny"] = pd.to_datetime(out["calendar_day_ny"])

    # Cast ML features to float32 for lighter training payload
    feature_cols = [c for c in out.columns if c.startswith("f_") or c.startswith("sent_")]
    out[feature_cols] = out[feature_cols].astype(np.float32)

    return out


def main() -> None:
    final_data = load_pickle(FINAL_DATA)
    earning_dates = load_pickle(EARNING_DATES)

    cfg = BuildConfig(lookback_days=LOOKBACK_DAYS, price_col=PRICE_COL)
    table = build_event_table(
        final_data=final_data,
        earning_dates=earning_dates,
        cfg=cfg,
        drop_unknown_timing=not KEEP_UNKNOWN_TIMING,
    )

    print("Built event table:", table.shape)
    if len(table) > 0:
        print("Date range:", table["earnings_day"].min(), "→", table["earnings_day"].max())
        print("Event types:\n", table["event_type"].value_counts(dropna=False))
        print("Label balance (up=1):\n", table["label_up"].value_counts(normalize=True).rename("share"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    suffix = OUT.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(OUT, index=False)
    elif suffix == ".csv":
        table.to_csv(OUT, index=False)
    elif suffix == ".pkl":
        with open(OUT, "wb") as f:
            pickle.dump(table, f)
    else:
        raise ValueError("Unsupported output extension. Use .parquet, .csv, or .pkl")


if __name__ == "__main__":
    main()
