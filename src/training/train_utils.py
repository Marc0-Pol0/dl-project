"""
Shared utilities for training and evaluating earnings-event prediction models.

This module provides reusable, model-agnostic helpers used across all training scripts 
(logistic regression, XGBoost, MLP, LSTM, Transformer).

Responsibilities:
- Reading the event-level dataset (Parquet / CSV)
- Selecting feature columns by prefix (e.g. "sent_", "f_")
- Enforcing time ordering and time-based train / validation / test splits
- Extracting (X, y) matrices from event tables
- Computing and printing standard binary-classification metrics

Design principles:
- No model-specific logic (pure utilities)
- Time-aware splits only (no random shuffling)
- Safe defaults for imbalanced binary classification
- Reusable across classical ML and neural models

Expected input format:
- One row per earnings event
- A binary label column (default: "label_up")
- A date column defining temporal order (default: "earnings_day")
- Feature columns prefixed by configurable strings (default: "sent_", "f_")
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
    split_date: str = "2025-05-01"
    val_tail_frac: float = 0.15


def read_table(path: Path) -> pd.DataFrame:
    """Read a table from a Parquet or CSV file."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported input extension. Use .parquet or .csv")


def pick_feature_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    """Pick feature columns from df based on given prefixes."""
    feats: list[str] = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            feats.append(c)
    return feats


def ensure_sorted_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ensure the date column is in datetime format and sort the DataFrame by this column."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return d.sort_values(date_col).reset_index(drop=True)


def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and test based on date_col and split_date.
    Train: rows with date_col < split_date
    Test:  rows with date_col >= split_date
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    split_ts = pd.Timestamp(split_date)
    train = d[d[date_col] < split_ts].copy()
    test = d[d[date_col] >= split_ts].copy()
    return train, test


def time_val_split(train: pd.DataFrame, date_col: str, tail_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train into train and val based on tail_frac of unique dates in date_col.
    The last tail_frac fraction of unique dates are used for validation.
    """
    if tail_frac <= 0.0 or len(train) == 0:
        return train, train.iloc[0:0].copy()

    dates = np.array(sorted(pd.to_datetime(train[date_col]).unique()))
    if len(dates) <= 1:
        return train, train.iloc[0:0].copy()

    k = max(1, int(round(len(dates) * (1.0 - tail_frac))))
    cutoff = dates[min(k - 1, len(dates) - 1)]
    tr = train[pd.to_datetime(train[date_col]) <= cutoff].copy()
    va = train[pd.to_datetime(train[date_col]) > cutoff].copy()
    return tr, va


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict[str, float]:
    """Evaluate binary classification predictions and print metrics."""
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    ll = float(log_loss(y_true, np.vstack([1.0 - y_prob, y_prob]).T, labels=[0, 1]))
    bs = float(brier_score_loss(y_true, y_prob))

    print(f"accuracy: {acc:.4f}")
    print(f"auc:      {auc:.4f}")
    print(f"logloss:  {ll:.4f}")
    print(f"brier:    {bs:.4f}")
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print("classification_report:\n", classification_report(y_true, y_pred, digits=4))

    return {"accuracy": acc, "auc": auc, "logloss": ll, "brier": bs}


def print_split_sizes(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print the sizes of the full, train, val, and test DataFrames."""
    print("rows:", len(df), "train:", len(train), "val:", len(val), "test:", len(test))


def get_xy(df: pd.DataFrame, feature_cols: Sequence[str], label_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract feature matrix X and label vector y from DataFrame."""
    x = df.loc[:, list(feature_cols)]
    y = df[label_col].astype(int).to_numpy()
    return x, y
