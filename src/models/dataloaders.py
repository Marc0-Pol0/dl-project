import os
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "./data/trainable/"
EA_FILENAME = "earning_dates_500.pkl"
FEATURE_FILENAME = "final_data_500.pkl"
EA_PARQUET = "earning_dates_500.parquet"
FEATURE_PARQUET = "final_data_500.parquet"
SEQUENCE_LENGTH = 30  # days preceding the EA

# Label thresholds. PRICE_CHANGE_THRESHOLD is the fixed cutoff (0.03 = submitted
# version; robustness.py sweeps 0.01/0.02/0.05). If VOLATILITY_SCALED_LABEL, the
# cutoff is VOLATILITY_MULTIPLIER x trailing 30-day pre-event return volatility.
PRICE_CHANGE_THRESHOLD = 0.03
VOLATILITY_SCALED_LABEL = False
VOLATILITY_MULTIPLIER = 1.0

# --- Feature groups ----------------------------------------------------------
# Used both for the sentiment ablation and for the single-modality ablations
# (price-only / fundamentals-only / sentiment-only).
SENTIMENT_COLS = ["positive", "neutral", "negative"]
PRICE_COLS = ["adj_price", "MA3", "MA6"]
FUNDAMENTAL_COLS = [
    "Dividend_Yield", "Net_Margin", "Gross_Margin", "ROE", "ROA",
    "Debt_to_Equity", "eps_basic", "assets", "shldrs_eq", "inven",
    "cash_st", "debt", "net_debt", "net_inc", "oper_cf",
]

FEATURE_GROUPS = {
    "all": PRICE_COLS + FUNDAMENTAL_COLS + SENTIMENT_COLS,
    "no_sentiment": PRICE_COLS + FUNDAMENTAL_COLS,
    "price_only": PRICE_COLS,
    "fundamentals_only": FUNDAMENTAL_COLS,
    "sentiment_only": SENTIMENT_COLS,
}

# Temporal split. TEST_SPLIT_POINT: fraction of the 365-day window for
# train+val. VAL_FRACTION: tail of train+val held out as validation.
# EMBARGO_DAYS: gap enforced at val/test boundaries. Default 0: on this sample
# the train/test window gap is already 37 days, while a 30-day embargo leaves
# 7 test events. robustness.py sweeps it.
TEST_SPLIT_POINT = 0.87
VAL_FRACTION = 0.20
EMBARGO_DAYS = 0

VALID_SPLITS = ("train", "val", "test")


def load_trainable_data():
    """
    Load the feature panel and the announcement dates.

    Parquet is used when present, pickle otherwise; the two paths yield identical
    datasets. Parquet avoids a NumPy-version incompatibility between the pickle
    format and the available PyTorch build (see convert_data.py).
    """
    feat_pq = os.path.join(DATA_DIR, FEATURE_PARQUET)
    ea_pq = os.path.join(DATA_DIR, EA_PARQUET)

    if os.path.exists(feat_pq) and os.path.exists(ea_pq):
        long = pd.read_parquet(feat_pq)
        feature_cols = [c for c in long.columns if c not in ("ticker", "date")]
        consolidated = {}
        for ticker, grp in long.groupby("ticker", sort=False):
            df = grp.set_index("date")[feature_cols].sort_index()
            df.index = pd.DatetimeIndex(df.index)
            consolidated[ticker] = df

        ea_long = pd.read_parquet(ea_pq)
        # Parquet may round-trip the column as tz-aware; drop the tz WITHOUT
        # converting, to match the local-time convention of the pickle path.
        if hasattr(ea_long["Earnings Date"].dtype, "tz") and ea_long["Earnings Date"].dtype.tz is not None:
            ea_long["Earnings Date"] = ea_long["Earnings Date"].dt.tz_localize(None)
        ea_dates = {
            ticker: grp[["Earnings Date"]].reset_index(drop=True)
            for ticker, grp in ea_long.groupby("ticker", sort=False)
        }
        return consolidated, ea_dates

    return (pd.read_pickle(os.path.join(DATA_DIR, FEATURE_FILENAME)),
            pd.read_pickle(os.path.join(DATA_DIR, EA_FILENAME)))


def get_split_boundaries(ea_dates: dict[str, pd.DataFrame]) -> dict[str, pd.Timestamp]:
    """Chronological boundaries of the train / val / test partitions."""
    all_ea_dates = []
    for ea_df in ea_dates.values():
        all_ea_dates.extend(ea_df["Earnings Date"].dropna().tolist())

    max_date = max(all_ea_dates)
    min_date = max_date - dt.timedelta(days=365)

    test_cutoff = min_date + (max_date - min_date) * TEST_SPLIT_POINT
    val_cutoff = min_date + (test_cutoff - min_date) * (1.0 - VAL_FRACTION)

    return {"min_date": min_date, "val_cutoff": val_cutoff,
            "test_cutoff": test_cutoff, "max_date": max_date}


def announcement_return(ticker: str, ea_date, consolidated_data: dict[str, pd.DataFrame]) -> float:
    """
    Return over the announcement window.

    Two-calendar-day window: adjusted close the day before the announcement to
    adjusted close the day after. Applied uniformly, without conditioning on
    pre-open / intraday / after-close timing. Non-trading days are forward-filled
    upstream, so the number of trading days spanned can vary.
    """
    df = consolidated_data[ticker]
    post = pd.to_datetime(ea_date) + dt.timedelta(days=1)
    pre = post - dt.timedelta(days=2)
    p_pre = df.loc[pre]["adj_price"]
    p_post = df.loc[post]["adj_price"]
    return float((p_post - p_pre) / p_pre)


def trailing_volatility(ticker: str, ea_date, consolidated_data: dict[str, pd.DataFrame],
                        lookback: int = SEQUENCE_LENGTH) -> float:
    """Std of daily returns over `lookback` days ending the day before the
    announcement (pre-event information only)."""
    df = consolidated_data[ticker]
    end = pd.to_datetime(ea_date) - dt.timedelta(days=1)
    start = end - dt.timedelta(days=lookback)
    prices = df.loc[start:end]["adj_price"]
    rets = prices.pct_change().dropna()
    return float(rets.std()) if len(rets) > 1 else float("nan")


def calculate_label(ticker: str, ea_date, consolidated_data: dict[str, pd.DataFrame]) -> int:
    r = announcement_return(ticker, ea_date, consolidated_data)

    if VOLATILITY_SCALED_LABEL:
        vol = trailing_volatility(ticker, ea_date, consolidated_data)
        threshold = VOLATILITY_MULTIPLIER * vol
        if not np.isfinite(threshold) or threshold <= 0:
            threshold = PRICE_CHANGE_THRESHOLD
    else:
        threshold = PRICE_CHANGE_THRESHOLD

    if r >= threshold:
        return 0   # UP
    if r <= -threshold:
        return 1   # DOWN
    return 2       # NEUTRAL


class StockDataset(Dataset):
    def __init__(
        self,
        consolidated_data: dict[str, pd.DataFrame],
        ea_dates: dict[str, pd.DataFrame],
        scaler=None,
        is_train: bool = True,
        feature_group: str = "all",
    ):
        self.consolidated_data = consolidated_data
        self.ea_dates = ea_dates
        self.scaler = scaler
        self.is_train = is_train

        if feature_group not in FEATURE_GROUPS:
            raise ValueError(f"feature_group must be one of {list(FEATURE_GROUPS)}, got {feature_group!r}")
        self.feature_group = feature_group
        self.feature_cols = FEATURE_GROUPS[feature_group]

        # Fail on an unknown/absent feature column rather than dropping it.
        if consolidated_data:
            have = set(next(iter(consolidated_data.values())).columns)
            missing = [c for c in self.feature_cols if c not in have]
            if missing:
                raise ValueError(
                    f"feature_group {feature_group!r} names columns absent from the data: {missing}"
                )

        self.samples = []  # (ticker, start_date, ea_date, label)
        # Tally dropped events by reason (reported per partition).
        self.dropped = {"ticker_absent": 0, "window_before_history": 0, "label_unavailable": 0}

        for ticker, ea_df in ea_dates.items():
            if ticker not in consolidated_data:
                self.dropped["ticker_absent"] += len(ea_df)
                continue
            available = {d.date() for d in consolidated_data[ticker].index}

            for ea_date in ea_df["Earnings Date"].tolist():
                ea_date = pd.to_datetime(ea_date, errors="coerce").tz_localize(None).normalize().date()
                start_date = ea_date - dt.timedelta(days=SEQUENCE_LENGTH)
                if start_date not in available:
                    self.dropped["window_before_history"] += 1
                    continue
                try:
                    label = calculate_label(ticker, ea_date, consolidated_data)
                except (KeyError, IndexError):
                    self.dropped["label_unavailable"] += 1
                    continue
                self.samples.append((ticker, start_date, ea_date, label))

        # Deterministic order (ticker, ea_date), independent of the storage
        # format's dictionary iteration order.
        self.samples.sort(key=lambda r: (r[0], r[2]))
        self.samples_df = pd.DataFrame(self.samples, columns=["ticker", "start_date", "ea_date", "label"])

        if self.is_train and self.scaler is None:
            self.scaler = StandardScaler()
            blocks = []
            for _, row in self.samples_df.iterrows():
                win = consolidated_data[row["ticker"]].loc[
                    row["start_date"] : row["ea_date"] - dt.timedelta(days=1)
                ]
                blocks.append(self._select(win).values)
            self.scaler.fit(np.concatenate(blocks, axis=0))

    def _select(self, df_window: pd.DataFrame) -> pd.DataFrame:
        # Filter, preserving the source column order. Selecting in FEATURE_GROUPS
        # order instead would permute the inputs relative to the submitted run;
        # the permutation is irrelevant in principle but changes which randomly
        # initialised weights meet which feature, and therefore the trajectory.
        keep = set(self.feature_cols)
        cols = [c for c in df_window.columns if c in keep]
        return df_window[cols]

    def class_counts(self) -> dict[int, int]:
        if self.samples_df.empty:
            return {0: 0, 1: 0, 2: 0}
        c = self.samples_df["label"].value_counts().to_dict()
        return {k: int(c.get(k, 0)) for k in (0, 1, 2)}

    def n_tickers(self) -> int:
        return int(self.samples_df["ticker"].nunique()) if not self.samples_df.empty else 0

    def tickers(self) -> set:
        return set(self.samples_df["ticker"]) if not self.samples_df.empty else set()

    def date_range(self):
        if self.samples_df.empty:
            return None, None
        return self.samples_df["ea_date"].min(), self.samples_df["ea_date"].max()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, start_date, ea_date, label = self.samples[idx]
        win = self.consolidated_data[ticker].loc[start_date : ea_date - dt.timedelta(days=1)]

        # Enforce a fixed window length; a gap in the daily index would
        # otherwise surface later as a collate error.
        if len(win) != SEQUENCE_LENGTH:
            raise ValueError(
                f"{ticker}: window {start_date}..{ea_date - dt.timedelta(days=1)} has "
                f"{len(win)} rows, expected {SEQUENCE_LENGTH}. Check the daily index."
            )

        X = self._select(win).values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def create_dataloader(
    batch_size: int,
    split: str = None,
    scaler: StandardScaler = None,
    feature_group: str = "all",
    use_sentiment: bool = None,
    is_train: bool = None,
    apply_embargo: bool = True,
    num_workers: int = 0,
    verbose: bool = True,
    **kwargs,
) -> tuple[DataLoader, StandardScaler]:
    """
    Build a DataLoader for one chronological partition.

    split:         'train' | 'val' | 'test'
    feature_group: key of FEATURE_GROUPS
    use_sentiment: legacy flag; True -> 'all', False -> 'no_sentiment'
    is_train:      legacy flag; True -> 'train', False -> 'test'
    apply_embargo: require an event's whole input window to post-date the boundary

    num_workers defaults to 0 so the shuffling RNG stream depends only on the
    seed, which keeps runs reproducible across machines.
    """
    if use_sentiment is not None and feature_group == "all":
        feature_group = "all" if use_sentiment else "no_sentiment"

    if split is None:
        if is_train is None:
            raise ValueError("Provide split='train'|'val'|'test' or the legacy is_train flag.")
        split = "train" if is_train else "test"
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    consolidated_data, ea_dates = load_trainable_data()

    for ticker, ea_df in ea_dates.items():
        parsed = pd.to_datetime(ea_df["Earnings Date"], errors="coerce")
        # Local wall-clock time, tz dropped without conversion (see load_trainable_data).
        if hasattr(parsed.dtype, "tz") and parsed.dtype.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        ea_df["Earnings Date"] = parsed.dt.normalize()
        ea_df.sort_values("Earnings Date", inplace=True)
        ea_dates[ticker] = ea_df

    b = get_split_boundaries(ea_dates)
    embargo = dt.timedelta(days=EMBARGO_DAYS) if apply_embargo else dt.timedelta(days=0)

    if split == "train":
        lo, hi = None, b["val_cutoff"]
    elif split == "val":
        lo, hi = b["val_cutoff"] + embargo, b["test_cutoff"]
    else:
        lo, hi = b["test_cutoff"] + embargo, None

    filtered = {}
    for ticker, ea_df in ea_dates.items():
        mask = pd.Series(True, index=ea_df.index)
        if lo is not None:
            mask &= ea_df["Earnings Date"] > lo
        if hi is not None:
            mask &= ea_df["Earnings Date"] <= hi
        sel = ea_df[mask].copy()
        if not sel.empty:
            filtered[ticker] = sel

    dataset = StockDataset(
        consolidated_data=consolidated_data,
        ea_dates=filtered,
        scaler=scaler,
        is_train=(split == "train"),
        feature_group=feature_group,
    )

    if verbose:
        c = dataset.class_counts()
        d0, d1 = dataset.date_range()
        n = max(len(dataset), 1)
        print(
            f"  [{split:5s}] events={len(dataset):4d}  firms={dataset.n_tickers():3d}  "
            f"{d0} -> {d1}  UP={c[0]} DOWN={c[1]} NEUTRAL={c[2]} ({100*c[2]/n:.1f}% neutral)"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        **kwargs,
    )
    return loader, dataset.scaler


def partition_report(feature_group: str = "all", apply_embargo: bool = True) -> pd.DataFrame:
    """
    Dataset summary: event counts, firm counts, date ranges, class distribution
    per partition, and firm overlap across partitions.
    """
    rows, ticker_sets, samples = [], {}, {}
    scaler = None
    for split in VALID_SPLITS:
        loader, sc = create_dataloader(
            batch_size=64, split=split, scaler=scaler, feature_group=feature_group,
            apply_embargo=apply_embargo, verbose=False,
        )
        if split == "train":
            scaler = sc
        ds = loader.dataset
        c = ds.class_counts()
        d0, d1 = ds.date_range()
        n = max(len(ds), 1)
        ticker_sets[split] = ds.tickers()
        samples[split] = ds.samples_df
        rows.append({
            "partition": split, "events": len(ds), "firms": ds.n_tickers(),
            "first_event": d0, "last_event": d1,
            "UP": c[0], "DOWN": c[1], "NEUTRAL": c[2],
            "pct_UP": round(100 * c[0] / n, 1),
            "pct_DOWN": round(100 * c[1] / n, 1),
            "pct_NEUTRAL": round(100 * c[2] / n, 1),
            "dropped_window_before_history": ds.dropped["window_before_history"],
            "dropped_label_unavailable": ds.dropped["label_unavailable"],
        })

    df = pd.DataFrame(rows)

    # Window-level separation: whether a later partition's input windows reach
    # back into an earlier one (event dates alone do not show this).
    span = {}
    for split in VALID_SPLITS:
        sdf = samples[split]
        if sdf.empty:
            continue
        span[split] = (sdf["start_date"].min(), sdf["ea_date"].max() - dt.timedelta(days=1))
    df.attrs["window_span"] = {k: (str(a), str(b)) for k, (a, b) in span.items()}

    if "train" in span and "test" in span:
        df.attrs["gap_train_window_end_to_first_test_window_days"] = (
            samples["test"]["start_date"].min() - span["train"][1]
        ).days
        df.attrs["test_windows_reaching_into_train_period"] = int(
            (samples["test"]["start_date"] <= span["train"][1]).sum()
        )
    if "val" in span and "test" in span:
        df.attrs["test_windows_reaching_into_val_period"] = int(
            (samples["test"]["start_date"] <= span["val"][1]).sum()
        )

    df.attrs["firms_train_and_test"] = len(ticker_sets["train"] & ticker_sets["test"])
    df.attrs["firms_train_and_val"] = len(ticker_sets["train"] & ticker_sets["val"])
    df.attrs["firms_in_all_three"] = len(
        ticker_sets["train"] & ticker_sets["val"] & ticker_sets["test"]
    )
    return df
