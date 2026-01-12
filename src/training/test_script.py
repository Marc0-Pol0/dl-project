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
PRICE_COL = "adj_price"

KEEP_UNKNOWN_TIMING = False
NY_TZ = "America/New_York"

NEUTRAL_RET_EPS = 0.03  # if |ret| <= eps, label is neutral


@dataclass(frozen=True)
class BuildConfig:
    lookback_trading_days: int = LOOKBACK_TRADING_DAYS
    price_col: str = PRICE_COL
    sentiment_cols: tuple[str, str, str] = ("positive", "negative", "neutral")
    neutral_ret_eps: float = NEUTRAL_RET_EPS

    include_snapshot_cols: tuple[str, ...] | None = None
    exclude_snapshot_cols: tuple[str, ...] = ("positive", "negative", "neutral")
    exclude_snapshot_prefixes: tuple[str, ...] = ("sent_",)

# Rest of main:
    # print("Built event table:", table.shape)
    # if not table.empty:
    #     print("Date range:", table["earnings_day"].min(), "→", table["earnings_day"].max())
    #     print("Event types:\n", table["event_type"].value_counts(dropna=False))

    #     counts = table["label"].value_counts(dropna=False)
    #     shares = (counts / counts.sum()).rename("share")
    #     print(f"Label repartition (eps={cfg.neutral_ret_eps:.6f}):\n", counts)
    #     print("Label shares:\n", shares)
    #     print("Binary label_up balance:\n", table["label_up"].value_counts(normalize=True).rename("share"))

    # OUT.parent.mkdir(parents=True, exist_ok=True)
    # ext = OUT.suffix.lower()
    # if ext == ".parquet":
    #     table.to_parquet(OUT, index=False)
    # elif ext == ".csv":
    #     table.to_csv(OUT, index=False)
    # elif ext == ".pkl":
    #     with open(OUT, "wb") as f:
    #         pickle.dump(table, f)
    # else:
    #     raise ValueError("Unsupported output extension. Use .parquet, .csv, or .pkl")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
    

final_data = load_pickle(FINAL_DATA)
earning_dates = load_pickle(EARNING_DATES)

tickers = sorted(set(final_data) & set(earning_dates))
ticker_examples = tickers[:5]
ticker_ex = tickers[0]

def ensure_tradingday_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a sorted, unique, tz-naive, date-normalized DatetimeIndex and drop weekends."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")

    out = df.sort_index()

    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_localize(None)

    out = out.copy()
    out.index = out.index.normalize()

    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]

    out = out[out.index.dayofweek < 5]
    return out

print(final_data[ticker_ex].index)
print(ensure_tradingday_index(final_data[ticker_ex]).index)



# Build event table 
""""
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
                if event_type == "UNKNOWN" and not keep_unknown_timing:
                    dropped_unknown += 1
                    continue

                calendar_day = pd.Timestamp(ts_ny.date()).normalize()

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

                if event_type == "BMO":
                    t_minus_1 = previous_trading_day(idx, event_day)
                    t_minus_2 = None if t_minus_1 is None else previous_trading_day(idx, t_minus_1)
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

                sent_feats = sentiment_aggregates(df, idx, last_included, cfg)
                snap = snapshot_features(df, feature_day, cfg)

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
                    "label_neutral": int(label_neutral),
                    "label_up": int(label_up),
                }
                row.update(sent_feats)
                row.update(snap)
                rows.append(row)

        except Exception as ex:
            print(f"[WARN] ticker={t} failed: {type(ex).__name__}: {ex}")

    if dropped_nat:
        pass

    out = pd.DataFrame(rows)

    print(
        "[INFO] dropped: "
        f"NaT={dropped_nat}, UNKNOWN={dropped_unknown}, outside_window={dropped_outside}, "
        f"mapping_failed={dropped_mapping}, no_return={dropped_return}, leakage_unsatisfied={dropped_exposure}"
    )

    if out.empty:
        return out

    out = out.sort_values(["earnings_day", "ticker"]).reset_index(drop=True)

    for c in ("earnings_ts_ny", "earnings_day", "feature_day", "calendar_day_ny"):
        out[c] = pd.to_datetime(out[c])

    out["event_type"] = out["event_type"].astype("category")
    out["label"] = out["label"].astype("category")

    sent_days_col = "sent_days_used"
    float_cols = [
        c for c in out.columns if c.startswith("f_") or (c.startswith("sent_") and c != sent_days_col)
    ]
    if float_cols:
        out[float_cols] = out[float_cols].astype(np.float32)
    if sent_days_col in out.columns:
        out[sent_days_col] = out[sent_days_col].astype(np.int32)

    return out"""





# def to_ny_timestamp(x: Any) -> pd.Timestamp:
#     if pd.isna(x):
#         return pd.NaT
#     ts = pd.to_datetime(x)
#     if ts.tzinfo is None:
#         return ts.tz_localize(NY_TZ)
#     return ts.tz_convert(NY_TZ)


# def classify_event_type(ts_ny: pd.Timestamp) -> str:
#     mins = int(ts_ny.hour) * 60 + int(ts_ny.minute)
#     if mins < 9 * 60 + 30:
#         return "BMO"
#     if mins >= 16 * 60:
#         return "AMC"
#     return "UNKNOWN"


# def map_trading_day_or_next(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
#     d = pd.Timestamp(day).normalize()
#     pos = int(trading_index.searchsorted(d, side="left"))
#     if pos >= len(trading_index):
#         return None
#     return trading_index[pos]


# def map_trading_day_or_prev(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
#     d = pd.Timestamp(day).normalize()
#     pos = int(trading_index.searchsorted(d, side="left"))
#     if pos < len(trading_index) and trading_index[pos] == d:
#         return trading_index[pos]
#     prev_pos = pos - 1
#     if prev_pos < 0:
#         return None
#     return trading_index[prev_pos]


# def previous_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
#     d = pd.Timestamp(day).normalize()
#     pos = int(trading_index.searchsorted(d, side="left"))
#     prev_pos = pos - 1
#     if prev_pos < 0:
#         return None
#     return trading_index[prev_pos]


# def next_trading_day(trading_index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
#     d = pd.Timestamp(day).normalize()
#     pos = int(trading_index.searchsorted(d, side="left"))
#     if pos < len(trading_index) and trading_index[pos] == d:
#         nxt_pos = pos + 1
#     else:
#         nxt_pos = pos
#     if nxt_pos >= len(trading_index):
#         return None
#     return trading_index[nxt_pos]


# def linreg_slope(y: np.ndarray) -> float:
#     n = int(len(y))
#     if n < 2:
#         return float("nan")
#     x = np.arange(n, dtype=float)
#     y = y.astype(float)
#     vx = x - x.mean()
#     denom = float((vx * vx).sum())
#     if denom <= 0.0:
#         return float("nan")
#     return float((vx * (y - y.mean())).sum() / denom)


# def make_event_id(ticker: str, ts_ny: pd.Timestamp) -> str:
#     return f"{ticker}_{ts_ny.strftime('%Y%m%dT%H%M%S%z')}"


# def label_from_return(ret: float, eps: float) -> tuple[str, int, int]:
#     if not math.isfinite(ret) or abs(ret) <= eps:
#         return ("neutral", 1, 0)
#     if ret > eps:
#         return ("up", 0, 1)
#     return ("down", 0, 0)





# def select_last_n_trading_days(trading_index: pd.DatetimeIndex, end_day: pd.Timestamp, n: int) -> pd.DatetimeIndex:
#     d = pd.Timestamp(end_day).normalize()
#     if d not in trading_index:
#         return trading_index[:0]
#     end_pos = int(trading_index.get_loc(d))
#     start_pos = max(0, end_pos - n + 1)
#     return trading_index[start_pos : end_pos + 1]


# def sentiment_aggregates(
#     df: pd.DataFrame, trading_index: pd.DatetimeIndex, last_included_day: pd.Timestamp, cfg: BuildConfig
# ) -> dict[str, float | int]:
#     pos_col, neg_col, neu_col = cfg.sentiment_cols
#     missing = [c for c in (pos_col, neg_col, neu_col) if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing sentiment columns: {missing}")

#     days = select_last_n_trading_days(trading_index, last_included_day, cfg.lookback_trading_days)
#     out: dict[str, float | int] = {}

#     if len(days) == 0:
#         for name in ("mean", "std", "trend"):
#             out[f"sent_pos_{name}"] = float("nan")
#             out[f"sent_neg_{name}"] = float("nan")
#             out[f"sent_neu_{name}"] = float("nan")
#         out["sent_pos_minus_neg_mean"] = float("nan")
#         out["sent_days_used"] = 0
#         return out

#     window = df.loc[days, [pos_col, neg_col, neu_col]]
#     out["sent_days_used"] = int(len(window))

#     for col, pref in ((pos_col, "pos"), (neg_col, "neg"), (neu_col, "neu")):
#         s = window[col].astype(float)
#         out[f"sent_{pref}_mean"] = float(s.mean())
#         out[f"sent_{pref}_std"] = float(s.std(ddof=0))
#         out[f"sent_{pref}_trend"] = linreg_slope(s.to_numpy())

#     out["sent_pos_minus_neg_mean"] = float(out["sent_pos_mean"]) - float(out["sent_neg_mean"])
#     return out


# def compute_event_return(
#     df: pd.DataFrame, event_day: pd.Timestamp, event_type: str, cfg: BuildConfig
# ) -> tuple[float, float, pd.Timestamp] | None:
#     if cfg.price_col not in df.columns:
#         raise KeyError(f"Missing price column '{cfg.price_col}'")

#     idx = df.index
#     d = pd.Timestamp(event_day).normalize()
#     if d not in idx:
#         return None

#     if event_type == "BMO":
#         prev_day = previous_trading_day(idx, d)
#         if prev_day is None:
#             return None
#         p0 = float(df.at[prev_day, cfg.price_col])
#         p1 = float(df.at[d, cfg.price_col])
#         if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
#             return None
#         ret = p1 / p0 - 1.0
#         return (float(ret), float(abs(ret)), prev_day)

#     if event_type == "AMC":
#         nxt_day = next_trading_day(idx, d)
#         if nxt_day is None:
#             return None
#         p0 = float(df.at[d, cfg.price_col])
#         p1 = float(df.at[nxt_day, cfg.price_col])
#         if not (math.isfinite(p0) and math.isfinite(p1)) or p0 == 0.0:
#             return None
#         ret = p1 / p0 - 1.0
#         return (float(ret), float(abs(ret)), d)

#     return None


# def snapshot_features(df: pd.DataFrame, feature_day: pd.Timestamp, cfg: BuildConfig) -> dict[str, object]:
#     d = pd.Timestamp(feature_day).normalize()
#     if d not in df.index:
#         return {}

#     if cfg.include_snapshot_cols is not None:
#         cols = [c for c in cfg.include_snapshot_cols if c in df.columns]
#     else:
#         exclude = set(cfg.exclude_snapshot_cols)
#         cols: list[str] = []
#         for c in df.columns:
#             if c in exclude:
#                 continue
#             if any(c.startswith(p) for p in cfg.exclude_snapshot_prefixes):
#                 continue
#             cols.append(c)

#     snap = df.loc[d, cols].to_dict()
#     return {f"f_{k}": v for k, v in snap.items()}
