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

C = 1.0
MAX_ITER = 10000
NO_BALANCED = False
RANDOM_STATE = 0


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
        class_weight=None if NO_BALANCED else "balanced",
        random_state=int(RANDOM_STATE),
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def run_split(name: str, pipe: Pipeline, part: pd.DataFrame, feature_cols: list[str]) -> None:
    """Run evaluation on a data split and print metrics."""
    if len(part) == 0:
        print(f"\n[{name}] empty")
        return

    x, y, class_names = get_xy(part, feature_cols, "label")

    print(f"\n[{name}]")
    proba = pipe.predict_proba(x)  # (n, K)
    eval_multiclass(y, proba, class_names=class_names)


def main() -> None:
    df = read_table(DATA)
    df = ensure_sorted_datetime(df, "earnings_day")
    feature_cols = pick_feature_columns(df, ("sent_", "f_"))

    train_all, test = time_split(df, "earnings_day", SPLIT_DATE)
    train, val = time_val_split(train_all, "earnings_day", VAL_TAIL_FRAC)

    print_split_sizes(df, "earnings_day", train, val, test)
    print("feature_cols:", len(feature_cols))
    print("label_col:", "label")

    pipe = build_logreg_pipeline()

    x_tr, y_tr, class_names = get_xy(train, feature_cols, "label")

    pipe.fit(x_tr, y_tr)

    run_split("train", pipe, train, feature_cols)
    run_split("val", pipe, val, feature_cols)
    run_split("test", pipe, test, feature_cols)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()
