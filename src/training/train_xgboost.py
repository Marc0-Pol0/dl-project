from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from train_utils import (
    ensure_sorted_datetime,
    eval_multiclass,
    get_xy,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)


DATA = Path("data/trainable/event_table_500.parquet")
OUT = Path("networks/xgb_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

RANDOM_STATE = 0
CLASS_ORDER = ["heavy_down", "down", "neutral", "up", "heavy_up"]

N_ESTIMATORS = 600
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
GAMMA = 0.0


def build_xgb_pipeline() -> Pipeline:
    pre = SimpleImputer(strategy="median")

    objective = "multi:softprob"

    clf = XGBClassifier(
        objective=objective,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        tree_method="hist",
    )

    return Pipeline(steps=[("impute", pre), ("clf", clf)])


def run_split(name: str, pipe: Pipeline, part: pd.DataFrame, feature_cols: list[str], model_name: str = "xgb") -> None:
    if len(part) == 0:
        print(f"\n[{name}] empty")
        return

    x, y, class_names = get_xy(part, feature_cols, "label", class_order=CLASS_ORDER)

    print(f"\n[{name}]")
    proba = pipe.predict_proba(x)  # (n, K)
    eval_multiclass(y, proba, class_names=class_names, model_name=model_name)


def main() -> None:
    df = read_table(DATA)

    df = ensure_sorted_datetime(df, "earnings_day")
    feature_cols = pick_feature_columns(df, ("sent_", "f_"))
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefixes: {('sent_', 'f_')}")

    train_all, test = time_split(df, SPLIT_DATE, date_col="earnings_day")
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col="earnings_day")

    print_split_sizes(df, train, val, test, date_col="earnings_day")
    print("feature_cols:", len(feature_cols))
    print("label_col:", "label")
    x_tr, y_tr, class_names = get_xy(train, feature_cols, "label", class_order=CLASS_ORDER)

    pipe = build_xgb_pipeline()
    pipe.fit(x_tr, y_tr)

    run_split("train", pipe, train, feature_cols, model_name="xgb_train")
    run_split("val", pipe, val, feature_cols, model_name="xgb_val")
    run_split("test", pipe, test, feature_cols, model_name="xgb_test")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "class_order": CLASS_ORDER,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()
