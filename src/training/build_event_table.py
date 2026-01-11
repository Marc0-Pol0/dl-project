"""
Build an earnings-event table (features + label) from:

1) final_data_500.pkl:
   dict[ticker] -> DataFrame indexed by trading-day DatetimeIndex (tz-naive preferred),
   with columns like positive/negative/neutral + fundamentals + adj_price + MA3/MA6.

2) earning_dates_500.pkl:
   dict[ticker] -> DataFrame with column "Earnings Date" (tz-aware, America/New_York).

Output:
  A single event-level DataFrame with one row per earnings event in the modeling window:
    - ticker, earnings_ts_ny, calendar_day_ny, earnings_day (mapped trading day), event_type (BMO/AMC), feature_day
    - return, label (down/neutral/up), label_neutral (binary), label_up (binary), abs_return
    - aggregated sentiment over a lookback window (last N trading days; mean/std/trend + pos-neg)
    - snapshot of selected per-day features from final_data at feature_day (prefixed "f_")
    - event_id (ticker + timestamp) for debugging

Leakage controls (trading-day accurate; defaults are conservative):
  - For BMO: exclude T-1 (day before earnings) => last included sentiment day is T-2 trading day
  - For AMC: exclude T (earnings day)          => last included sentiment day is T-1 trading day
  - Features used:
        BMO -> previous trading day (pre-announcement close)
        AMC -> same trading day (close before announcement)

Other controls:
  - Trading-day mapping uses event_type and date presence in index:
        BMO: map to trading day on date if available, else next trading day
        AMC: map to trading day on date if available, else previous trading day (otherwise drop)
  - Mapping returns None if outside coverage
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

LOOKBACK_TRADING_DAYS = 30
PRICE_COL = "adj_price"

KEEP_UNKNOWN_TIMING = False
NY_TZ = "America/New_York"

# Neutral labeling config:
NEUTRAL_RET_EPS = 0.01  # 1% band: |ret| <= eps => neutral


# -------------------------- helpers --------------------------

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame index is a sorted, timezone-naive DatetimeIndex,
    and restrict to (approx) trading days by dropping weekends.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")

    out = df.sort_index()

    # Make tz-naive if needed
    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_convert(None)

    # Drop duplicate dates if any (keep last by default)
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]

    # Drop weekends (dayofweek: Mon=0 ... Sun=6)
    out = out[out.index.dayofweek < 5]

    return out


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


def map_trading_day_or_next(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """If day is a trading day, return it; else return next trading day. None if out of range."""
    day0 = pd.Timestamp(day.date())
    pos = int(trading_index.searchsorted(day0, side="left"))
    if pos >= len(trading_index):
        return None
    return trading_index[pos]


def map_trading_day_or_prev(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """If day is a trading day, return it; else return previous trading day. None if out of range."""
    day0 = pd.Timestamp(day.date())
    pos = int(trading_index.searchsorted(day0, side="left"))
    if pos < len(trading_index) and trading_index[pos] == day0:
        return trading_index[pos]
    prev_pos = pos - 1
    if prev_pos < 0 or prev_pos >= len(trading_index):
        return None
    return trading_index[prev_pos]


def previous_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """Return the previous trading day before 'day', or None if none exists."""
    pos = int(trading_index.searchsorted(day, side="left"))
    prev_pos = pos - 1
    if prev_pos < 0:
        return None
    return trading_index[prev_pos]


def next_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    """Return the next trading day after 'day', or None if none exists."""
    pos = int(trading_index.searchsorted(day, side="left"))
    if pos < len(trading_index) and trading_index[pos] == day:
        nxt_pos = pos + 1
    else:
        nxt_pos = pos
    if nxt_pos >= len(trading_index):
        return None
    return trading_index[nxt_pos]


def linreg_slope(y: np.ndarray) -> float:
    """Slope of y on x where x = 0..n-1. Returns NaN if ill-posed."""
    n = int(len(y))
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    y = y.astype(float)
    vx = x - x.mean()
    denom = float((vx * vx).sum())
    if denom <= 0.0:
        return float("nan")
    return float((vx * (y - y.mean())).sum() / denom)


def event_id(ticker: str, ts_ny: pd.Timestamp) -> str:
    return f"{ticker}_{ts_ny.strftime('%Y%m%dT%H%M%S%z')}"


def label_from_return(ret: float, eps: float) -> tuple[str, int, int]:
    """
    3-class label based on return:
      - neutral if |ret| <= eps
      - up if ret > eps
      - down if ret < -eps
    Returns: (label, label_neutral, label_up)
    """
    if not math.isfinite(ret):
        return ("neutral", 1, 0)
    if abs(ret) <= eps:
        return ("neutral", 1, 0)
    if ret > eps:
        return ("up", 0, 1)
    return ("down", 0, 0)


# -------------------------- build config --------------------------

@dataclass(frozen=True)
class BuildConfig:
    lookback_trading_days: int = 30
    price_col: str = "adj_price"
    sentiment_cols: tuple[str, str, str] = ("positive", "negative", "neutral")

    # Neutral labeling
    neutral_ret_eps: float = 0.0025

    # Snapshot inclusion/exclusion
    include_snapshot_cols: tuple[str, ...] | None = None  # if set, use only these (must exist)
    exclude_snapshot_cols: tuple[str, ...] = ("positive", "negative", "neutral")  # always excluded by default
    exclude_snapshot_prefixes: tuple[str, ...] = ("sent_",)


# -------------------------- feature builders --------------------------

def select_last_n_trading_days(trading_index: pd.DatetimeIndex, end_day: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Return the last n trading days ending at end_day (inclusive). Empty if end_day not in index."""
    if end_day not in trading_index:
        return trading_index[:0]
    end_pos = int(trading_index.get_loc(end_day))
    start_pos = max(0, end_pos - n + 1)
    return trading_index[start_pos : end_pos + 1]


def sentiment_aggregates_trading_days(
    df: pd.DataFrame, trading_index: pd.DatetimeIndex, last_included_day: pd.Timestamp, cfg: BuildConfig
) -> dict[str, float | int]:
    """Aggregate sentiment over last N trading days ending at last_included_day (inclusive)."""
    pos_col, neg_col, neu_col = cfg.sentiment_cols
    for c in (pos_col, neg_col, neu_col):
        if c not in df.columns:
            raise KeyError(f"Missing sentiment column: {c}")

    days = select_last_n_trading_days(trading_index, last_included_day, cfg.lookback_trading_days)
    out: dict[str, float | int] = {}

    if len(days) == 0:
        for name in ("mean", "std", "trend"):
            out[f"sent_pos_{name}"] = float("nan")
            out[f"sent_neg_{name}"] = float("nan")
            out[f"sent_neu_{name}"] = float("nan")
        out["sent_pos_minus_neg_mean"] = float("nan")
        out["sent_days_used"] = 0
        return out

    window = df.loc[days, [pos_col, neg_col, neu_col]]
    out["sent_days_used"] = int(len(window))

    for col, pref in ((pos_col, "pos"), (neg_col, "neg"), (neu_col, "neu")):
        s = window[col].astype(float)
        out[f"sent_{pref}_mean"] = float(s.mean())
        out[f"sent_{pref}_std"] = float(s.std(ddof=0))
        out[f"sent_{pref}_trend"] = linreg_slope(s.to_numpy())

    out["sent_pos_minus_neg_mean"] = float(out["sent_pos_mean"]) - float(out["sent_neg_mean"])
    return out


def compute_event_return(
    df: pd.DataFrame, event_day: pd.Timestamp, event_type: str, cfg: BuildConfig
) -> tuple[float, float, pd.Timestamp] | None:
    """
    Returns (ret, abs_ret, feature_day), or None if not computable.

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
        return (float(ret), float(abs(ret)), prev_day)

    if event_type == "AMC":
        nxt_day = next_trading_day(idx, event_day)
        if nxt_day is None:
            return None
        p0 = float(df.at[event_day, cfg.price_col])
        p1 = float(df.at[nxt_day, cfg.price_col])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
            return None
        ret = p1 / p0 - 1.0
        return (float(ret), float(abs(ret)), event_day)

    return None


def snapshot_features(df: pd.DataFrame, feature_day: pd.Timestamp, cfg: BuildConfig) -> dict[str, object]:
    """Snapshot a controlled set of columns at feature_day, prefixed by f_."""
    if feature_day not in df.index:
        return {}

    if cfg.include_snapshot_cols is not None:
        cols = [c for c in cfg.include_snapshot_cols if c in df.columns]
    else:
        exclude = set(cfg.exclude_snapshot_cols)
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            if any(c.startswith(p) for p in cfg.exclude_snapshot_prefixes):
                continue
            cols.append(c)

    snap = df.loc[feature_day, cols].to_dict()
    return {f"f_{k}": v for k, v in snap.items()}


# -------------------------- main build --------------------------

def build_event_table(
    final_data: dict[str, pd.DataFrame],
    earning_dates: dict[str, pd.DataFrame],
    cfg: BuildConfig,
    drop_unknown_timing: bool = True,
) -> pd.DataFrame:
    tickers = sorted(set(final_data.keys()) & set(earning_dates.keys()))
    rows: list[dict[str, object]] = []

    dropped_nat = 0
    dropped_unknown = 0
    dropped_outside = 0
    dropped_mapping = 0
    dropped_return = 0
    dropped_exposure = 0  # cannot satisfy leakage rule (missing prev days)

    for t in tickers:
        try:
            df = ensure_datetime_index(final_data[t])
            if len(df) == 0:
                continue
            idx = df.index
            start_day = idx.min()
            end_day = idx.max()

            e = earning_dates[t]
            if "Earnings Date" not in e.columns:
                raise KeyError(f"{t}: earnings df missing 'Earnings Date' column")

            earn_ts_ny_series = e["Earnings Date"].map(to_ny_timestamp).dropna()
            earn_ts_ny_series = earn_ts_ny_series.sort_values()

            for ts_ny in earn_ts_ny_series:
                if pd.isna(ts_ny):
                    dropped_nat += 1
                    continue

                event_type = classify_event_type(ts_ny)
                if event_type == "UNKNOWN" and drop_unknown_timing:
                    dropped_unknown += 1
                    continue

                calendar_day = pd.Timestamp(ts_ny.date())  # tz-naive NY calendar day at midnight

                if calendar_day < start_day or calendar_day > end_day:
                    dropped_outside += 1
                    continue

                if event_type == "BMO":
                    event_day = map_trading_day_or_next(idx, calendar_day)
                elif event_type == "AMC":
                    event_day = map_trading_day_or_prev(idx, calendar_day)
                else:
                    dropped_mapping += 1
                    continue

                if event_day is None or event_day < start_day or event_day > end_day:
                    dropped_mapping += 1
                    continue

                ret_info = compute_event_return(df, event_day, event_type, cfg)
                if ret_info is None:
                    dropped_return += 1
                    continue
                ret, abs_ret, feature_day = ret_info

                label, label_neutral, label_up = label_from_return(ret, cfg.neutral_ret_eps)

                # Leakage: last included sentiment day in trading days
                if event_type == "BMO":
                    t_minus_1 = previous_trading_day(idx, event_day)
                    if t_minus_1 is None:
                        dropped_exposure += 1
                        continue
                    t_minus_2 = previous_trading_day(idx, t_minus_1)
                    if t_minus_2 is None:
                        dropped_exposure += 1
                        continue
                    last_included = t_minus_2
                else:  # AMC
                    t_minus_1 = previous_trading_day(idx, event_day)
                    if t_minus_1 is None:
                        dropped_exposure += 1
                        continue
                    last_included = t_minus_1

                sent_feats = sentiment_aggregates_trading_days(df, idx, last_included, cfg)
                snap = snapshot_features(df, feature_day, cfg)

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
                    "label": label,                       # {down, neutral, up}
                    "label_neutral": int(label_neutral),  # binary
                    "label_up": int(label_up),            # binary (for backward compat)
                }
                row.update(sent_feats)
                row.update(snap)
                rows.append(row)

        except Exception as ex:
            print(f"[WARN] ticker={t} failed with: {type(ex).__name__}: {ex}")

    out = pd.DataFrame(rows)

    print(
        "[INFO] dropped: "
        f"NaT={dropped_nat}, UNKNOWN={dropped_unknown}, outside_window={dropped_outside}, "
        f"mapping_failed={dropped_mapping}, no_return={dropped_return}, leakage_unsatisfied={dropped_exposure}"
    )

    if len(out) == 0:
        return out

    out = out.sort_values(["earnings_day", "ticker"]).reset_index(drop=True)

    # Cast timestamps
    out["earnings_ts_ny"] = pd.to_datetime(out["earnings_ts_ny"])
    out["earnings_day"] = pd.to_datetime(out["earnings_day"])
    out["feature_day"] = pd.to_datetime(out["feature_day"])
    out["calendar_day_ny"] = pd.to_datetime(out["calendar_day_ny"])

    # Types
    out["event_type"] = out["event_type"].astype("category")
    out["label"] = out["label"].astype("category")

    # Cast ML features
    sent_days_col = "sent_days_used"
    float_feature_cols = [
        c
        for c in out.columns
        if c.startswith("f_") or (c.startswith("sent_") and c != sent_days_col)
    ]
    out[float_feature_cols] = out[float_feature_cols].astype(np.float32)

    if sent_days_col in out.columns:
        out[sent_days_col] = out[sent_days_col].astype(np.int32)

    return out


def main() -> None:
    final_data = load_pickle(FINAL_DATA)
    earning_dates = load_pickle(EARNING_DATES)

    cfg = BuildConfig(
        lookback_trading_days=LOOKBACK_TRADING_DAYS,
        price_col=PRICE_COL,
        sentiment_cols=("positive", "negative", "neutral"),
        neutral_ret_eps=NEUTRAL_RET_EPS,
        include_snapshot_cols=None,
        exclude_snapshot_cols=("positive", "negative", "neutral"),
        exclude_snapshot_prefixes=("sent_",),
    )

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

        # 3-class label repartition (counts + shares)
        counts = table["label"].value_counts(dropna=False)
        shares = (counts / counts.sum()).rename("share")
        print(f"Label repartition (eps={cfg.neutral_ret_eps:.6f}):\n", counts)
        print("Label shares:\n", shares)

        # If you still want the old binary balance for reference:
        print("Binary label_up balance:\n", table["label_up"].value_counts(normalize=True).rename("share"))

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
