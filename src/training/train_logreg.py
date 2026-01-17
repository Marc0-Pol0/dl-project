from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
OUT = Path("networks/logreg_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

C = 5.0
MAX_ITER = 1000
BALANCED = True

RANDOM_STATE = 0
CLASS_ORDER = ["down", "neutral", "up"]


def build_logreg_pipeline() -> Pipeline:
    pre = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    clf = LogisticRegression(
        C=float(C),
        solver="lbfgs",
        max_iter=int(MAX_ITER),
        class_weight="balanced" if BALANCED else None,
        random_state=int(RANDOM_STATE)
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def run_split(
        name: str, 
        pipe: Pipeline, 
        part: pd.DataFrame, 
        feature_cols: list[str], 
        class_order: list[str], 
        model_name: str = "logreg",
    ) -> None:
    """Run evaluation on a data split and print metrics."""
    if len(part) == 0:
        print(f"\n[{name}] empty")
        return

    x, y, class_names = get_xy(part, feature_cols, "label", class_order=class_order, nan_policy="raise")

    print(f"\n[{name}]")
    proba = pipe.predict_proba(x)  # (n, K)
    eval_multiclass(y, proba, class_names=class_names, model_name=model_name)


def main() -> None:
    df = read_table(DATA)
    df = ensure_sorted_datetime(df, "earnings_day")

    feature_cols = pick_feature_columns(df, ("sent_", "f_"))

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found with prefixes ('sent_', 'f_').")

    train_all, test = time_split(df, SPLIT_DATE, date_col="earnings_day")
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col="earnings_day")

    print_split_sizes(df, train, val, test, date_col="earnings_day")
    print("feature_cols:", len(feature_cols))
    print("label_col:", "label")
    print("class_order:", CLASS_ORDER)

    pipe = build_logreg_pipeline()

    x_tr, y_tr, class_names = get_xy(train, feature_cols, "label", class_order=CLASS_ORDER, nan_policy="raise")

    if list(class_names) != list(CLASS_ORDER):
        raise RuntimeError(f"class_names={class_names} != CLASS_ORDER={CLASS_ORDER}")

    pipe.fit(x_tr, y_tr)

    run_split("train", pipe, train, feature_cols, CLASS_ORDER, model_name="logreg_train")
    run_split("val", pipe, val, feature_cols, CLASS_ORDER, model_name="logreg_val")
    run_split("test", pipe, test, feature_cols, CLASS_ORDER, model_name="logreg_test")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "class_order": CLASS_ORDER,
        "date_col": "earnings_day",
        "split_date": SPLIT_DATE,
        "val_tail_frac": float(VAL_TAIL_FRAC),
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()
