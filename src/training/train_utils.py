"""
Shared utilities for training and evaluating earnings-event prediction models.

Reads the event-level dataset (Parquet / CSV)
Selects feature columns by prefix (e.g. "sent_", "f_")
Enforces time ordering and time-based train / validation / test splits
Extracts (X, y) matrices from event tables
Computes and prints standard binary-classification metrics
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class SplitConfig:
    label_col: str = "label_up"
    date_col: str = "earnings_day"
    feature_prefixes: tuple[str, ...] = ("sent_", "f_")
    # Train/Test split: < split_date is train; >= split_date is test
    split_date: str = "2025-05-01"
    # Validation is the last tail fraction of unique dates inside the train period
    val_tail_frac: float = 0.15


# -------------------------- IO --------------------------

def read_table(path: Path) -> pd.DataFrame:
    """Read a table from a Parquet or CSV file."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension '{suf}'. Use .parquet or .csv")


# -------------------------- schema / hygiene --------------------------

def ensure_sorted_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ensure date_col is datetime64 and sort ascending by date_col."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    return d.sort_values(date_col).reset_index(drop=True)


# -------------------------- feature selection --------------------------

def pick_feature_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    """Pick feature columns based on prefixes; returns a sorted, stable list."""
    prefixes = tuple(prefixes)
    feats = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    feats.sort()
    return feats


# -------------------------- time splits --------------------------

def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and test based on date_col and split_date.
      Train: date_col < split_date
      Test:  date_col >= split_date
    """
    d = ensure_sorted_datetime(df, date_col)
    split_ts = pd.Timestamp(split_date)
    train = d[d[date_col] < split_ts].copy()
    test = d[d[date_col] >= split_ts].copy()
    return train, test


def time_val_split(train: pd.DataFrame, date_col: str, tail_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into (train, val) using the last tail_frac fraction of unique dates as validation."""
    if len(train) == 0 or tail_frac <= 0.0:
        return train, train.iloc[0:0].copy()

    d = ensure_sorted_datetime(train, date_col)
    dates = pd.to_datetime(d[date_col]).dropna().unique()
    dates = np.array(sorted(dates))
    if len(dates) <= 1:
        return d, d.iloc[0:0].copy()

    # Number of unique dates reserved for validation
    n_val_dates = max(1, int(round(len(dates) * float(tail_frac))))
    n_val_dates = min(n_val_dates, len(dates) - 1)  # keep at least 1 date for training
    cutoff = dates[-n_val_dates - 1]

    tr = d[pd.to_datetime(d[date_col]) <= cutoff].copy()
    va = d[pd.to_datetime(d[date_col]) > cutoff].copy()
    return tr, va


def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Convenience wrapper:
      - sort by time
      - pick features by prefix
      - split into train/val/test by time

    Returns (train, val, test, feature_cols).
    """
    d = ensure_sorted_datetime(df, cfg.date_col)

    feature_cols = pick_feature_columns(d, cfg.feature_prefixes)
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for prefixes={cfg.feature_prefixes}")

    train_full, test = time_split(d, cfg.date_col, cfg.split_date)
    train, val = time_val_split(train_full, cfg.date_col, cfg.val_tail_frac)
    return train, val, test, feature_cols


def print_split_sizes(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print sizes of full/train/val/test plus date ranges."""
    def _rng(x: pd.DataFrame, col: str) -> str:
        if len(x) == 0 or col not in x.columns:
            return "∅"
        a = pd.to_datetime(x[col]).min()
        b = pd.to_datetime(x[col]).max()
        return f"{a.date()} → {b.date()}"

    print("rows:", len(df), "| train:", len(train), "| val:", len(val), "| test:", len(test))
    if len(df) > 0:
        print("date ranges:",
              "train:", _rng(train, "earnings_day"),
              "| val:", _rng(val, "earnings_day"),
              "| test:", _rng(test, "earnings_day"))


# -------------------------- X/y extraction --------------------------

def get_xy(df: pd.DataFrame, feature_cols: Sequence[str], label_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y.
    Returns numpy arrays for easy use with sklearn / torch.
    """
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")
    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    y = df[label_col].astype(int).to_numpy(copy=True)
    return x, y


# -------------------------- evaluation --------------------------

def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5, eps: float = 1e-7) -> dict[str, float]:
    """
    Evaluate binary classification predictions and print metrics.

    y_prob is P(y=1). Clipped to [eps, 1-eps] for numeric stability.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    y_pred = (y_prob >= float(thresh)).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    ll = float(log_loss(y_true, np.c_[1.0 - y_prob, y_prob], labels=[0, 1]))
    bs = float(brier_score_loss(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print(f"accuracy: {acc:.4f}")
    print(f"auc:      {auc:.4f}")
    print(f"logloss:  {ll:.4f}")
    print(f"brier:    {bs:.4f}")
    print("confusion_matrix:\n", cm)
    print("classification_report:\n", rep)

    return {"accuracy": acc, "auc": auc, "logloss": ll, "brier": bs}
