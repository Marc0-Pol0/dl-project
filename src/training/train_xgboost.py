"""
Train and evaluate an XGBoost baseline on the earnings-event dataset.

Pipeline:
1. Load the event-level earnings dataset
2. Sort by time
3. Select numerical features by prefix ("sent_", "f_")
4. Impute missing values (median); XGBoost does not require standardization
5. Train an XGBoost model (binary:logistic)
6. Evaluate performance on train / val / test splits
7. Save the trained model bundle for later inference or comparison
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
OUT = Path("networks/xgb_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

# -------------------------- model hyperparams --------------------------

RANDOM_STATE = 0

N_ESTIMATORS = 600
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
GAMMA = 0.0

# If True, set scale_pos_weight = (#neg / #pos) on the training split
AUTO_SCALE_POS_WEIGHT = True


@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int = 600
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    scale_pos_weight: float = 1.0
    random_state: int = 0
    n_jobs: int = -1
    tree_method: str = "hist"
    eval_metric: str = "logloss"


def build_xgb_pipeline(cfg: XGBConfig) -> Pipeline:
    """
    Build an XGBoost pipeline:
      - median imputation (keeps everything numeric)
      - XGBClassifier (binary:logistic)
    """
    pre = SimpleImputer(strategy="median")

    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(cfg.n_estimators),
        max_depth=int(cfg.max_depth),
        learning_rate=float(cfg.learning_rate),
        subsample=float(cfg.subsample),
        colsample_bytree=float(cfg.colsample_bytree),
        reg_lambda=float(cfg.reg_lambda),
        min_child_weight=float(cfg.min_child_weight),
        gamma=float(cfg.gamma),
        scale_pos_weight=float(cfg.scale_pos_weight),
        random_state=int(cfg.random_state),
        n_jobs=int(cfg.n_jobs),
        eval_metric=str(cfg.eval_metric),
        tree_method=str(cfg.tree_method),
    )

    return Pipeline(steps=[("impute", pre), ("clf", clf)])


def run_split(name: str, pipe: Pipeline, part: pd.DataFrame, feature_cols: list[str], label_col: str) -> None:
    """Run evaluation on a data split and print metrics."""
    if len(part) == 0:
        print(f"\n[{name}] empty")
        return

    x, y = get_xy(part, feature_cols, label_col)
    prob = pipe.predict_proba(x)[:, 1]
    print(f"\n[{name}]")
    eval_binary(y, prob, thresh=0.5)


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight = (#neg / #pos)."""
    y = np.asarray(y_train).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


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

    x_tr, y_tr = get_xy(train, feature_cols, split_cfg.label_col)

    spw = compute_scale_pos_weight(y_tr) if AUTO_SCALE_POS_WEIGHT else 1.0
    if AUTO_SCALE_POS_WEIGHT:
        print(f"scale_pos_weight (neg/pos) on train: {spw:.4f}")

    xgb_cfg = XGBConfig(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )

    pipe = build_xgb_pipeline(xgb_cfg)
    pipe.fit(x_tr, y_tr)

    run_split("train", pipe, train, feature_cols, split_cfg.label_col)
    run_split("val", pipe, val, feature_cols, split_cfg.label_col)
    run_split("test", pipe, test, feature_cols, split_cfg.label_col)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "split_cfg": split_cfg,
        "xgb_cfg": xgb_cfg,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()
