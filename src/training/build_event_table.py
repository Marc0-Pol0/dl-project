import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FINAL_DATA = Path("./data/trainable/final_data_500.pkl")
EARNING_DATES = Path("./data/trainable/earning_dates_500.pkl")
OUT = Path("./data/trainable/event_table_500.parquet")

LOOKBACK_TRADING_DAYS = 30

NEUTRAL_RET_EPS = 0.02  # |ret| <= eps -> neutral


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_weekday_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_index().copy()
    out.index = out.index.normalize()
    out = out[out.index.dayofweek < 5]
    return out


def to_ny_timestamp(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x)
    if ts.tzinfo is None:
        return ts.tz_localize("America/New_York")
    return ts.tz_convert("America/New_York")


def classify_event_type(ts_ny: pd.Timestamp) -> str:
    mins = int(ts_ny.hour) * 60 + int(ts_ny.minute)
    if mins < 9 * 60 + 30:
        return "BMO"
    if mins >= 16 * 60:
        return "AMC"
    return "INTRADAY"


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
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = y.astype(float)
    vx = x - x.mean()
    denom = float((vx * vx).sum())
    if denom == 0.0:
        return 0.0
    return float((vx * (y - y.mean())).sum() / denom)


def make_event_id(ticker: str, ts_ny: pd.Timestamp) -> str:
    return f"{ticker}_{ts_ny.strftime('%Y%m%dT%H%M%S%z')}"


def label_3way_from_return(ret: float, neutral_eps: float) -> str:
    if abs(ret) <= neutral_eps:
        return "neutral"
    elif ret > 0:
        return "up"
    else:
        return "down"


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
    window = df.loc[days, [pos_col, neg_col, neu_col]]

    out: dict[str, float | int] = {"sent_days_used": int(len(window))}
    if window.empty:
        for pref in ("pos", "neg", "neu"):
            out[f"sent_{pref}_mean"] = 0.0
            out[f"sent_{pref}_std"] = 0.0
            out[f"sent_{pref}_trend"] = 0.0
        out["sent_pos_minus_neg_mean"] = 0.0
        return out

    for col, pref in ((pos_col, "pos"), (neg_col, "neg"), (neu_col, "neu")):
        s = window[col].astype(float)
        out[f"sent_{pref}_mean"] = float(s.mean())
        out[f"sent_{pref}_std"] = float(s.std(ddof=0))
        out[f"sent_{pref}_trend"] = linreg_slope(s.to_numpy())

    out["sent_pos_minus_neg_mean"] = float(out["sent_pos_mean"]) - float(out["sent_neg_mean"])
    return out


def sentiment_sequence(
    df: pd.DataFrame, trading_index: pd.DatetimeIndex, last_included_day: pd.Timestamp, n: int
) -> dict[str, object]:
    pos_col, neg_col, neu_col = ("positive", "negative", "neutral")
    days = select_last_n_trading_days(trading_index, last_included_day, n)

    pos = np.zeros(n, dtype=np.float32)
    neg = np.zeros(n, dtype=np.float32)
    neu = np.zeros(n, dtype=np.float32)
    pn = np.zeros(n, dtype=np.float32)

    if len(days) > 0:
        w = df.loc[days, [pos_col, neg_col, neu_col]].astype(float)
        vals_pos = w[pos_col].to_numpy(dtype=np.float32)
        vals_neg = w[neg_col].to_numpy(dtype=np.float32)
        vals_neu = w[neu_col].to_numpy(dtype=np.float32)
        vals_pn = vals_pos - vals_neg

        pos[-len(days) :] = vals_pos
        neg[-len(days) :] = vals_neg
        neu[-len(days) :] = vals_neu
        pn[-len(days) :] = vals_pn

    return {
        "sent_seq_len": int(len(days)),
        "sent_seq_pos": pos.tolist(),
        "sent_seq_neg": neg.tolist(),
        "sent_seq_neu": neu.tolist(),
        "sent_seq_pn": pn.tolist(),
    }


def compute_return_close_to_close(df: pd.DataFrame, day0: pd.Timestamp, day1: pd.Timestamp) -> float | None:
    if day0 not in df.index or day1 not in df.index:
        return None
    p0 = df.at[day0, "adj_price"]
    p1 = df.at[day1, "adj_price"]
    if pd.isna(p0) or pd.isna(p1):
        return None
    if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
        return None
    return float(p1 / p0 - 1.0)


@dataclass(frozen=True)
class EventSpec:
    event_type: str
    event_day: pd.Timestamp
    feature_day: pd.Timestamp
    last_sentiment_day: pd.Timestamp
    ret_day0: pd.Timestamp
    ret_day1: pd.Timestamp


def make_event_spec(idx: pd.DatetimeIndex, calendar_day: pd.Timestamp, event_type: str) -> EventSpec | None:
    """
    Pre-announcement policy:
    - BMO: prev close -> event-day close; features/sentiment up to prev day.
    - AMC: event-day close -> next close; features/sentiment up to prev day.
    - INTRADAY: treat like AMC for return; use sentiment up to prev day to avoid same-day leakage.
    """
    d = pd.Timestamp(calendar_day).normalize()

    if event_type == "BMO":
        event_day = map_trading_day_or_next(idx, d)
        if event_day is None:
            return None
        prev_day = previous_trading_day(idx, event_day)
        if prev_day is None:
            return None
        ret_day0, ret_day1 = prev_day, event_day
        feature_day = prev_day
        last_sentiment_day = prev_day
        return EventSpec(event_type, event_day, feature_day, last_sentiment_day, ret_day0, ret_day1)

    if event_type == "AMC":
        event_day = map_trading_day_or_prev(idx, d)
        if event_day is None:
            return None
        prev_day = previous_trading_day(idx, event_day)
        if prev_day is None:
            return None
        next_day = next_trading_day(idx, event_day)
        if next_day is None:
            return None
        ret_day0, ret_day1 = event_day, next_day
        feature_day = prev_day
        last_sentiment_day = prev_day
        return EventSpec(event_type, event_day, feature_day, last_sentiment_day, ret_day0, ret_day1)

    else:  # INTRADAY
        event_day = map_trading_day_or_prev(idx, d)
        if event_day is None:
            return None
        next_day = next_trading_day(idx, event_day)
        if next_day is None:
            return None
        ret_day0, ret_day1 = event_day, next_day
        prev_day = previous_trading_day(idx, event_day)
        if prev_day is None:
            return None
        feature_day = prev_day
        last_sentiment_day = prev_day
        return EventSpec(event_type, event_day, feature_day, last_sentiment_day, ret_day0, ret_day1)


def snapshot_features(df: pd.DataFrame, feature_day: pd.Timestamp) -> dict[str, object] | None:
    d = pd.Timestamp(feature_day).normalize()
    if d not in df.index:
        return None

    exclude = {"positive", "negative", "neutral"}
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
    final_data: dict[str, pd.DataFrame], earning_dates: dict[str, pd.DataFrame], include_sent_sequence: bool = True
) -> pd.DataFrame:
    tickers = sorted(set(final_data) & set(earning_dates))
    rows: list[dict[str, object]] = []

    dropped_outside = 0
    dropped_intraday = 0
    dropped_unmappable = 0
    dropped_missing_prices = 0
    dropped_missing_features = 0

    for t in tickers:
        df = ensure_weekday_index(final_data[t])
        if df.empty:
            continue

        idx = df.index
        start_day = idx.min()
        end_day = idx.max()

        e = earning_dates[t]
        if "Earnings Date" not in e.columns:
            print(f"[WARN] ticker={t} missing 'Earnings Date' column")
            continue

        earn_ts_ny = e["Earnings Date"].map(to_ny_timestamp).dropna().sort_values()

        for ts_ny in earn_ts_ny:
            calendar_day = pd.Timestamp(ts_ny.date()).normalize()

            if calendar_day < start_day or calendar_day > end_day:
                dropped_outside += 1
                continue

            event_type = classify_event_type(ts_ny)
            if event_type == "INTRADAY":
                dropped_intraday += 1
                continue

            spec = make_event_spec(idx, calendar_day, event_type)
            if spec is None:
                dropped_unmappable += 1
                continue

            ret = compute_return_close_to_close(df, spec.ret_day0, spec.ret_day1)
            if ret is None:
                dropped_missing_prices += 1
                continue

            snap = snapshot_features(df, spec.feature_day)
            if snap is None:
                dropped_missing_features += 1
                continue

            sent_feats = sentiment_aggregates(df, idx, spec.last_sentiment_day)

            row: dict[str, object] = {
                "event_id": make_event_id(t, ts_ny),
                "ticker": t,
                "earnings_ts_ny": ts_ny,
                "event_type": spec.event_type,
                "calendar_day_ny": calendar_day,
                "earnings_day": spec.event_day,
                "feature_day": spec.feature_day,
                "sent_last_included_day": spec.last_sentiment_day,
                "return": float(ret),
                "abs_return": float(abs(ret)),
                "label": label_3way_from_return(ret, NEUTRAL_RET_EPS),
            }
            row.update(sent_feats)
            row.update(snap)

            if include_sent_sequence:
                row.update(sentiment_sequence(df, idx, spec.last_sentiment_day, LOOKBACK_TRADING_DAYS))

            rows.append(row)

    out = pd.DataFrame(rows)
    print(
        "[INFO] dropped:",
        f"outside_window={dropped_outside},",
        f"intraday={dropped_intraday},",
        f"unmappable={dropped_unmappable},",
        f"missing_prices={dropped_missing_prices},",
        f"missing_features={dropped_missing_features}",
    )

    if out.empty:
        return out

    out = out.sort_values(["earnings_day", "ticker"]).reset_index(drop=True)

    for c in ("earnings_ts_ny", "earnings_day", "feature_day", "calendar_day_ny", "sent_last_included_day"):
        out[c] = pd.to_datetime(out[c])

    out["event_type"] = out["event_type"].astype("category")
    out["label"] = out["label"].astype("category")

    sent_days_col = "sent_days_used"

    float_cols: list[str] = []
    for c in out.columns:
        if c.startswith("f_") or (c.startswith("sent_") and c != sent_days_col):
            if pd.api.types.is_numeric_dtype(out[c]):
                float_cols.append(c)

    if float_cols:
        out[float_cols] = out[float_cols].astype(np.float32)

    if sent_days_col in out.columns and pd.api.types.is_numeric_dtype(out[sent_days_col]):
        out[sent_days_col] = out[sent_days_col].astype(np.int32)

    if "sent_seq_len" in out.columns:
        out["sent_seq_len"] = out["sent_seq_len"].astype(np.int32)

    return out


def main() -> None:
    final_data = load_pickle(FINAL_DATA)
    earning_dates = load_pickle(EARNING_DATES)

    table = build_event_table(final_data=final_data, earning_dates=earning_dates, include_sent_sequence=True)

    print("Built event table:", table.shape)
    if not table.empty:
        print("Date range:", table["earnings_day"].min(), "→", table["earnings_day"].max())
        print("Event types:\n", table["event_type"].value_counts(dropna=False))

        counts = table["label"].value_counts(dropna=False)
        shares = (counts / counts.sum()).rename("share")
        print(f"Label distribution (neutral={NEUTRAL_RET_EPS:.6f}):\n", counts)
        print("Label shares:\n", shares)

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
