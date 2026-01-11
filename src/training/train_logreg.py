"""
Train and evaluate a logistic regression baseline on the earnings-event dataset.

Pipeline:
1. Load the event-level earnings dataset
2. Sort by time
3. Select numerical features by prefix ("sent_", "f_")
4. Perform time-based train / validation / test splits
5. Apply preprocessing (median imputation + standardization)
6. Train a logistic regression classifier (optionally class-balanced)
7. Evaluate performance on train / val / test splits
8. Save a trained model bundle for later inference or comparison

Outputs:
- Printed evaluation metrics (accuracy, AUC, log loss, Brier score)
- Serialized model bundle (.joblib) including pipeline and configuration
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_utils import (
    SplitConfig,
    ensure_sorted_datetime,
    eval_binary,
    get_xy,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)

# -------------------------- paths / split --------------------------

DATA = Path("data/trainable/event_table_500.parquet")
OUT = Path("networks/logreg_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

# -------------------------- model hyperparams --------------------------

C = 1.0
MAX_ITER = 10000
NO_BALANCED = False
RANDOM_STATE = 0


@dataclass(frozen=True)
class LogRegConfig:
    c: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"
    random_state: int = 0


def build_logreg_pipeline(cfg: LogRegConfig) -> Pipeline:
    pre = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    clf = LogisticRegression(
        l1_ratio=0,
        C=float(cfg.c),
        solver="lbfgs",
        max_iter=int(cfg.max_iter),
        class_weight=cfg.class_weight,
        random_state=int(cfg.random_state),
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def run_split(name: str, pipe: Pipeline, part: pd.DataFrame, feature_cols: list[str], label_col: str) -> None:
    """Run evaluation on a data split and print metrics."""
    if len(part) == 0:
        print(f"\n[{name}] empty")
        return

    x, y = get_xy(part, feature_cols, label_col)
    prob = pipe.predict_proba(x)[:, 1]
    print(f"\n[{name}]")
    eval_binary(y, prob, thresh=0.5)


def main() -> None:
    split_cfg = SplitConfig(split_date=SPLIT_DATE, val_tail_frac=VAL_TAIL_FRAC)

    df = read_table(DATA)
    if split_cfg.label_col not in df.columns:
        raise KeyError(f"Missing label column: {split_cfg.label_col}")
    if split_cfg.date_col not in df.columns:
        raise KeyError(f"Missing date column: {split_cfg.date_col}")


    df = ensure_sorted_datetime(df, split_cfg.date_col)
    feature_cols = pick_feature_columns(df, split_cfg.feature_prefixes)
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefixes: {split_cfg.feature_prefixes}")

    train_all, test = time_split(df, split_cfg.date_col, split_cfg.split_date)
    train, val = time_val_split(train_all, split_cfg.date_col, split_cfg.val_tail_frac)

    print_split_sizes(df, train, val, test)
    print("feature_cols:", len(feature_cols))

    lr_cfg = LogRegConfig(
        c=C, max_iter=MAX_ITER, class_weight=None if NO_BALANCED else "balanced", random_state=RANDOM_STATE
    )

    pipe = build_logreg_pipeline(lr_cfg)

    x_tr, y_tr = get_xy(train, feature_cols, split_cfg.label_col)
    pipe.fit(x_tr, y_tr)

    run_split("train", pipe, train, feature_cols, split_cfg.label_col)
    run_split("val", pipe, val, feature_cols, split_cfg.label_col)
    run_split("test", pipe, test, feature_cols, split_cfg.label_col)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "split_cfg": split_cfg,
        "logreg_cfg": lr_cfg,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()
