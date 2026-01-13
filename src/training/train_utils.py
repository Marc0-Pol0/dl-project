from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension '{suf}'. Use .parquet or .csv")


def ensure_sorted_datetime(df: pd.DataFrame, date_col: str = "earnings_day") -> pd.DataFrame:
    d = df.copy()
    if date_col not in d.columns:
        raise KeyError(f"Missing date column: {date_col}")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    return d.sort_values(date_col).reset_index(drop=True)


def pick_feature_columns(
    df: pd.DataFrame, prefixes: Iterable[str] = ("sent_", "f_"), exclude_prefixes: Iterable[str] = ("sent_seq_",),
) -> list[str]:
    """Pick feature columns whose names start with any of the given prefixes."""
    prefixes_t = tuple(prefixes)
    exclude_t = tuple(exclude_prefixes)
    feats = [c for c in df.columns if any(c.startswith(p) for p in prefixes_t)]
    feats = [c for c in feats if not any(c.startswith(ep) for ep in exclude_t)]

    feats.sort()
    return feats


def _coerce_feature_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Build a dense float32 matrix from the selected feature columns."""
    X_parts: list[np.ndarray] = []
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        X_parts.append(arr.reshape(-1, 1))
    if not X_parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(X_parts, axis=1)


def time_split(df: pd.DataFrame, split_date: str, date_col: str = "earnings_day") -> tuple[pd.DataFrame, pd.DataFrame]:
    d = ensure_sorted_datetime(df, date_col)
    split_ts = pd.Timestamp(split_date)
    train = d[d[date_col] < split_ts].copy()
    test = d[d[date_col] >= split_ts].copy()
    return train, test


def time_val_split(
        train: pd.DataFrame, tail_frac: float, date_col: str = "earnings_day"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into (train, val) by taking the last `tail_frac` of unique dates as validation."""
    if len(train) == 0 or tail_frac <= 0.0:
        return train, train.iloc[0:0].copy()

    d = ensure_sorted_datetime(train, date_col)

    dt = pd.to_datetime(d[date_col], errors="coerce").dropna()
    if dt.empty:
        return d, d.iloc[0:0].copy()

    dates = np.array(sorted(dt.unique()))
    if len(dates) <= 1:
        return d, d.iloc[0:0].copy()

    n_val_dates = max(1, int(round(len(dates) * float(tail_frac))))
    n_val_dates = min(n_val_dates, len(dates) - 1)
    cutoff = dates[-n_val_dates - 1]

    mask = pd.to_datetime(d[date_col]) <= cutoff
    tr = d[mask].copy()
    va = d[~mask].copy()
    return tr, va


def make_splits(
    df: pd.DataFrame,
    split_date: str,
    val_tail_frac: float,
    date_col: str = "earnings_day",
    feature_prefixes: Sequence[str] = ("sent_", "f_"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    d = ensure_sorted_datetime(df, date_col)

    feature_cols = pick_feature_columns(d, prefixes=feature_prefixes)
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for prefixes={tuple(feature_prefixes)}")

    train_full, test = time_split(d, split_date, date_col=date_col)
    train, val = time_val_split(train_full, val_tail_frac, date_col=date_col)
    return train, val, test, feature_cols


def print_split_sizes(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, date_col: str = "earnings_day"
) -> None:
    def _rng(x: pd.DataFrame, col: str) -> str:
        if len(x) == 0 or col not in x.columns:
            return "∅"
        a = pd.to_datetime(x[col]).min()
        b = pd.to_datetime(x[col]).max()
        return f"{a.date()} → {b.date()}"

    print("rows:", len(df), "| train:", len(train), "| val:", len(val), "| test:", len(test))
    if len(df) > 0:
        print(
            "date ranges:",
            "train:", _rng(train, date_col),
            "| val:", _rng(val, date_col),
            "| test:", _rng(test, date_col),
        )


def encode_labels(y: pd.Series, class_order: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Encode labels into int indices [0..K-1] and return (y_int, class_names)."""
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_integer_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Numeric label column contains NaNs after coercion.")
        uniq = np.array(sorted(pd.unique(y_num.astype(int))))
        mapping = {int(v): i for i, v in enumerate(uniq)}
        y_int = y_num.astype(int).map(mapping).to_numpy()
        class_names = [str(int(v)) for v in uniq]
        return y_int.astype(int), class_names

    if pd.api.types.is_float_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Float label column contains NaNs after coercion.")
        if not np.allclose(y_num.to_numpy(), np.round(y_num.to_numpy()), atol=0, rtol=0):
            raise ValueError("Float label column is not integral-valued; refusing to encode as classes.")
        y_ints = y_num.round().astype(int)
        uniq = np.array(sorted(pd.unique(y_ints)))
        mapping = {int(v): i for i, v in enumerate(uniq)}
        y_int = y_ints.map(mapping).to_numpy()
        class_names = [str(int(v)) for v in uniq]
        return y_int.astype(int), class_names

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
    label_col: str = "label",
    class_order: Sequence[str] | None = None,
    nan_policy: str = "raise",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    X = _coerce_feature_matrix(df, feature_cols)

    if nan_policy not in {"raise", "zero"}:
        raise ValueError("nan_policy must be 'raise' or 'zero'")

    if nan_policy == "raise":
        if np.isnan(X).any() or np.isinf(X).any():
            bad = np.where(~np.isfinite(X))
            raise ValueError(
                f"Found non-finite values in X at indices (row,col) like: {list(zip(bad[0][:5], bad[1][:5]))}"
            )
    else:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    y_int, class_names = encode_labels(df[label_col], class_order=class_order)
    return X, y_int, class_names


def eval_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Sequence[str] | None = None,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Multiclass evaluation.

    y_proba: shape (n, K), rows sum ~ 1.
    Prints: accuracy, logloss, multiclass Brier score, confusion matrix, classification report.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D (n,K), got shape {y_proba.shape}")
    n, k = y_proba.shape
    if n != len(y_true):
        raise ValueError(f"y_true length {len(y_true)} != y_proba rows {n}")

    if class_names is not None and len(class_names) != k:
        raise ValueError(f"class_names length {len(class_names)} != number of classes {k}")

    y_proba = np.clip(y_proba, eps, 1.0 - eps)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    y_pred = y_proba.argmax(axis=1)

    labels = list(range(k))
    target_names = list(class_names) if class_names is not None else [str(i) for i in labels]

    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, y_proba, labels=labels))

    onehot = np.eye(k, dtype=float)[y_true]
    brier_mc = float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    print(f"accuracy: {acc:.4f}")
    print(f"logloss:  {ll:.4f}")
    print(f"brier_mc: {brier_mc:.4f}")
    print("confusion_matrix:\n", cm)
    print("classification_report:\n", rep)

    return {"accuracy": acc, "logloss": ll, "brier_mc": brier_mc}
