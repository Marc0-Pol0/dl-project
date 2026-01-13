import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FINAL_DATA = Path("./data/trainable/final_data_500.pkl")
EARNING_DATES = Path("./data/trainable/earning_dates_500.pkl")
OUT = Path("./data/trainable/event_table_500.parquet")

LOOKBACK_TRADING_DAYS = 30
NEUTRAL_RET_EPS = 0.03  # if |ret| <= eps, label is neutral


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_tradingday_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_index()
    out = out.copy()
    out.index = out.index.normalize()
    out = out[out.index.dayofweek < 5]
    return out


def to_ny_timestamp(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        return ts.tz_localize("America/New_York")
    else:
        return ts.tz_convert("America/New_York")


def classify_event_type(ts_ny: pd.Timestamp) -> str:
    mins = int(ts_ny.hour) * 60 + int(ts_ny.minute)
    if 0 < mins < 9 * 60 + 30:
        return "BMO"
    elif mins >= 16 * 60:
        return "AMC"
    else:
        return "UNKNOWN"


def map_trading_day_or_next(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    d = pd.Timestamp(day).normalize()
    pos = int(trading_index.searchsorted(d, side="left"))
    if pos >= len(trading_index):
        return None
    return trading_index[pos]


def map_trading_day_or_prev(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    d = pd.Timestamp(day).normalize()
    pos = int(trading_index.searchsorted(d, side="left"))
    if pos < len(trading_index) and trading_index[pos] == d:
        return trading_index[pos]
    prev_pos = pos - 1
    if prev_pos < 0:
        return None
    return trading_index[prev_pos]


def previous_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    d = pd.Timestamp(day).normalize()
    pos = int(trading_index.searchsorted(d, side="left"))
    prev_pos = pos - 1
    if prev_pos < 0:
        return None
    return trading_index[prev_pos]


def next_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    d = pd.Timestamp(day).normalize()
    pos = int(trading_index.searchsorted(d, side="left"))
    if pos < len(trading_index) and trading_index[pos] == d:
        nxt_pos = pos + 1
    else:
        nxt_pos = pos
    if nxt_pos >= len(trading_index):
        return None
    return trading_index[nxt_pos]


def linreg_slope(y: np.ndarray) -> float:
    n = int(len(y))
    x = np.arange(n, dtype=float)
    y = y.astype(float)
    vx = x - x.mean()
    denom = float((vx * vx).sum())
    return float((vx * (y - y.mean())).sum() / denom)


def make_event_id(ticker: str, ts_ny: pd.Timestamp) -> str:
    return f"{ticker}_{ts_ny.strftime('%Y%m%dT%H%M%S%z')}"


def label_from_return(ret: float, eps: float) -> str:
    if ret > eps:
        return "up"
    elif ret < -eps:
        return "down"
    else:
        return "neutral"

def select_last_n_trading_days(trading_index: pd.DatetimeIndex, end_day: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    d = pd.Timestamp(end_day).normalize()
    if d not in trading_index:
        return trading_index[:0]
    end_pos = int(trading_index.get_loc(d))
    start_pos = max(0, end_pos - n + 1)
    return trading_index[start_pos : end_pos + 1]


def sentiment_aggregates(
    df: pd.DataFrame, trading_index: pd.DatetimeIndex, last_included_day: pd.Timestamp
) -> dict[str, float | int]:
    pos_col, neg_col, neu_col = ("positive", "negative", "neutral")

    days = select_last_n_trading_days(trading_index, last_included_day, LOOKBACK_TRADING_DAYS)
    out: dict[str, float | int] = {}

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
    df: pd.DataFrame, event_day: pd.Timestamp, event_type: str
) -> tuple[float, float, pd.Timestamp] | None:
    idx = df.index
    d = pd.Timestamp(event_day).normalize()

    if event_type == "BMO":
        prev_day = previous_trading_day(idx, d)
        if prev_day is None:
            return None
        p0 = float(df.at[prev_day, "adj_price"])
        p1 = float(df.at[d, "adj_price"])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
            return None
        ret = p1 / p0 - 1.0
        return (float(ret), float(abs(ret)), prev_day)

    if event_type == "AMC":
        nxt_day = next_trading_day(idx, d)
        if nxt_day is None:
            return None
        p0 = float(df.at[d, "adj_price"])
        p1 = float(df.at[nxt_day, "adj_price"])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
            return None
        ret = p1 / p0 - 1.0
        return (float(ret), float(abs(ret)), d)

    return None


def snapshot_features(df: pd.DataFrame, feature_day: pd.Timestamp) -> dict[str, object]:
    d = pd.Timestamp(feature_day).normalize()
    
    exclude = set(("positive", "negative", "neutral"))
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("sent_"):
            continue
        cols.append(c)

    snap = df.loc[d, cols].to_dict()
    return {f"f_{k}": v for k, v in snap.items()}


def build_event_table(
    final_data: dict[str, pd.DataFrame], earning_dates: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    tickers = sorted(set(final_data) & set(earning_dates))
    rows: list[dict[str, object]] = []

    dropped_nat = 0
    dropped_unknown = 0
    dropped_outside = 0

    for t in tickers:
        try:
            df = ensure_tradingday_index(final_data[t])
            if df.empty:
                continue
            idx = df.index
            start_day = idx.min()
            end_day = idx.max()

            e = earning_dates[t]
            if "Earnings Date" not in e.columns:
                raise KeyError(f"{t}: missing 'Earnings Date' column")

            earn_ts_ny = e["Earnings Date"].map(to_ny_timestamp).dropna().sort_values()

            for ts_ny in earn_ts_ny:
                event_type = classify_event_type(ts_ny)
                if event_type == "UNKNOWN":
                    dropped_unknown += 1
                    continue

                calendar_day = pd.Timestamp(ts_ny.date()).normalize()

                if calendar_day < start_day or calendar_day > end_day:
                    dropped_outside += 1
                    continue

                if event_type == "BMO":
                    event_day = map_trading_day_or_next(idx, calendar_day)
                else:  # AMC
                    event_day = map_trading_day_or_prev(idx, calendar_day)
                
                ret_info = compute_event_return(df, event_day, event_type)
                ret, abs_ret, feature_day = ret_info

                label = label_from_return(ret, NEUTRAL_RET_EPS)

                if event_type == "BMO":
                    t_minus_1 = previous_trading_day(idx, event_day)
                    t_minus_2 = None if t_minus_1 is None else previous_trading_day(idx, t_minus_1)
                    last_included = t_minus_2
                else:  # AMC
                    t_minus_1 = previous_trading_day(idx, event_day)
                    last_included = t_minus_1

                sent_feats = sentiment_aggregates(df, idx, last_included)
                snap = snapshot_features(df, feature_day)

                row: dict[str, object] = {
                    "event_id": make_event_id(t, ts_ny),
                    "ticker": t,
                    "earnings_ts_ny": ts_ny,
                    "event_type": event_type,
                    "calendar_day_ny": calendar_day,
                    "earnings_day": event_day,
                    "feature_day": feature_day,
                    "return": float(ret),
                    "abs_return": float(abs_ret),
                    "label": label,
                }
                row.update(sent_feats)
                row.update(snap)
                rows.append(row)

        except Exception as ex:
            print(f"[WARN] ticker={t} failed: {type(ex).__name__}: {ex}")

    if dropped_nat:
        pass

    out = pd.DataFrame(rows)

    print(f"[INFO] dropped: NaT={dropped_nat}, UNKNOWN={dropped_unknown}, outside_window={dropped_outside}")

    out = out.sort_values(["earnings_day", "ticker"]).reset_index(drop=True)

    for c in ("earnings_ts_ny", "earnings_day", "feature_day", "calendar_day_ny"):
        out[c] = pd.to_datetime(out[c])

    out["event_type"] = out["event_type"].astype("category")
    out["label"] = out["label"].astype("category")

    sent_days_col = "sent_days_used"
    float_cols = [c for c in out.columns if c.startswith("f_") or (c.startswith("sent_") and c != sent_days_col)]
    if float_cols:
        out[float_cols] = out[float_cols].astype(np.float32)
    if sent_days_col in out.columns:
        out[sent_days_col] = out[sent_days_col].astype(np.int32)

    return out


def main() -> None:
    final_data = load_pickle(FINAL_DATA)
    earning_dates = load_pickle(EARNING_DATES)

    table = build_event_table(final_data=final_data, earning_dates=earning_dates)

    print("Built event table:", table.shape)
    if not table.empty:
        print("Date range:", table["earnings_day"].min(), "→", table["earnings_day"].max())
        print("Event types:\n", table["event_type"].value_counts(dropna=False))

        counts = table["label"].value_counts(dropna=False)
        shares = (counts / counts.sum()).rename("share")
        print(f"Label repartition (eps={NEUTRAL_RET_EPS:.6f}):\n", counts)
        print("Label shares:\n", shares)
        table["label_up"] = (table["label"] == "up").astype(np.int8)
        print("Binary label_up balance:\n", table["label_up"].value_counts(normalize=True).rename("share"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ext = OUT.suffix.lower()
    if ext == ".parquet":
        table.to_parquet(OUT, index=False)
    elif ext == ".csv":
        table.to_csv(OUT, index=False)
    elif ext == ".pkl":
        with open(OUT, "wb") as f:
            pickle.dump(table, f)
    else:
        raise ValueError("Unsupported output extension. Use .parquet, .csv, or .pkl")


if __name__ == "__main__":
    main()
