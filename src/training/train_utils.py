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
    # For 3-class from the new builder, use "label".
    # For old behavior, use "label_up".
    label_col: str = "label"
    date_col: str = "earnings_day"
    feature_prefixes: tuple[str, ...] = ("sent_", "f_")
    split_date: str = "2025-05-01"
    val_tail_frac: float = 0.15

    # For multiclass string labels (only used when label_col == "label" or dtype is non-numeric)
    # Must match the builder's label names.
    class_order: tuple[str, ...] = ("down", "neutral", "up")


# -------------------------- IO --------------------------

def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension '{suf}'. Use .parquet or .csv")


# -------------------------- schema / hygiene --------------------------

def ensure_sorted_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    return d.sort_values(date_col).reset_index(drop=True)


# -------------------------- feature selection --------------------------

def pick_feature_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    prefixes = tuple(prefixes)
    feats = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    feats.sort()
    return feats


# -------------------------- time splits --------------------------

def time_split(df: pd.DataFrame, date_col: str, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = ensure_sorted_datetime(df, date_col)
    split_ts = pd.Timestamp(split_date)
    train = d[d[date_col] < split_ts].copy()
    test = d[d[date_col] >= split_ts].copy()
    return train, test


def time_val_split(train: pd.DataFrame, date_col: str, tail_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train) == 0 or tail_frac <= 0.0:
        return train, train.iloc[0:0].copy()

    d = ensure_sorted_datetime(train, date_col)
    dates = pd.to_datetime(d[date_col]).dropna().unique()
    dates = np.array(sorted(dates))
    if len(dates) <= 1:
        return d, d.iloc[0:0].copy()

    n_val_dates = max(1, int(round(len(dates) * float(tail_frac))))
    n_val_dates = min(n_val_dates, len(dates) - 1)
    cutoff = dates[-n_val_dates - 1]

    tr = d[pd.to_datetime(d[date_col]) <= cutoff].copy()
    va = d[pd.to_datetime(d[date_col]) > cutoff].copy()
    return tr, va


def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    d = ensure_sorted_datetime(df, cfg.date_col)

    feature_cols = pick_feature_columns(d, cfg.feature_prefixes)
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for prefixes={cfg.feature_prefixes}")

    train_full, test = time_split(d, cfg.date_col, cfg.split_date)
    train, val = time_val_split(train_full, cfg.date_col, cfg.val_tail_frac)
    return train, val, test, feature_cols


def print_split_sizes(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
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

def encode_labels(
    y: pd.Series,
    class_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode labels into int indices [0..K-1] and return (y_int, class_names).
    - If y is numeric/bool already, keeps unique sorted values as class names.
    - If y is strings/categorical, uses class_order if provided; otherwise uses sorted uniques.
    """
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_integer_dtype(y) or pd.api.types.is_float_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Numeric label column contains NaNs after coercion.")
        uniq = np.array(sorted(pd.unique(y_num.astype(int))))
        mapping = {int(v): i for i, v in enumerate(uniq)}
        y_int = y_num.astype(int).map(mapping).to_numpy()
        class_names = [str(int(v)) for v in uniq]
        return y_int.astype(int), class_names

    # string/categorical path
    y_str = y.astype(str)
    if class_order is None:
        classes = sorted(pd.unique(y_str))
    else:
        classes = list(class_order)

    cat = pd.Categorical(y_str, categories=classes, ordered=True)
    if (cat.codes < 0).any():
        bad = sorted(set(y_str[cat.codes < 0]))
        raise ValueError(f"Found labels not in class_order: {bad}")
    return cat.codes.astype(int), list(classes)


def get_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    class_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    x = df.loc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    y_int, class_names = encode_labels(df[label_col], class_order=class_order)
    return x, y_int, class_names


# -------------------------- evaluation --------------------------

def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5, eps: float = 1e-7) -> dict[str, float]:
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


def eval_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str] | None = None,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Multiclass evaluation.
    - y_proba: shape (n, K), rows sum ~ 1
    Prints: accuracy, logloss, confusion matrix, classification report.
    Also prints multiclass Brier score (mean over samples of sum_k (p_k - 1[y=k])^2).
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D (n,K), got shape {y_proba.shape}")
    n, k = y_proba.shape
    if n != len(y_true):
        raise ValueError(f"y_true length {len(y_true)} != y_proba rows {n}")

    y_proba = np.clip(y_proba, eps, 1.0 - eps)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_pred = y_proba.argmax(axis=1)

    labels = list(range(k))
    target_names = list(class_names) if class_names is not None else [str(i) for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, y_proba, labels=labels))

    # Multiclass Brier: mean ||p - onehot||^2
    onehot = np.eye(k, dtype=float)[y_true]
    brier_mc = float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0
    )

    print(f"accuracy: {acc:.4f}")
    print(f"logloss:  {ll:.4f}")
    print(f"brier_mc: {brier_mc:.4f}")
    print("confusion_matrix:\n", cm)
    print("classification_report:\n", rep)

    return {"accuracy": acc, "logloss": ll, "brier_mc": brier_mc}
