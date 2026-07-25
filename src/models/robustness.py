"""
Robustness checks. Kept separate from train.py because each block re-labels or
re-partitions the data and is slower; three seeds are used here.

Blocks:
  A. Label threshold sweep (+/-1/2/3/5%) plus a volatility-scaled cutoff.
     Reports class distribution as well as performance.
  B. Single-modality ablations: price / fundamentals / sentiment only, vs full.
  C. Embargo length sweep, reporting the test sample size at each length and
     evaluating only lengths that leave an evaluable test set.

Run after train.py:  python src/models/robustness.py
"""

import os
import json
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import dataloaders as DL
from dataloaders import create_dataloader
import train as T


SEEDS = [0, 1, 2]
FIGURES_DIR = "./src/figures"

THRESHOLD_GRID = [0.01, 0.02, 0.03, 0.05]
VOL_MULTIPLIERS = [1.0, 1.5]

MODELS = ["logreg", "lstm", "attention"]
MODALITIES = ["all", "no_sentiment", "price_only", "fundamentals_only", "sentiment_only"]


def evaluate(model_name: str, feature_group: str, device, seeds=SEEDS, apply_embargo=None) -> dict:
    """
    Train and evaluate one configuration under the currently configured labels.

    apply_embargo defaults to train.py's setting; block C overrides it.
    """
    if apply_embargo is None:
        apply_embargo = T.Config.APPLY_EMBARGO
    train_loader, scaler = create_dataloader(
        batch_size=T.Config.BATCH_SIZE, split="train", feature_group=feature_group,
        apply_embargo=apply_embargo, verbose=False,
    )
    val_loader, _ = create_dataloader(
        batch_size=T.Config.BATCH_SIZE, split="val", scaler=scaler, feature_group=feature_group,
        apply_embargo=apply_embargo, verbose=False,
    )
    test_loader, _ = create_dataloader(
        batch_size=T.Config.BATCH_SIZE, split="test", scaler=scaler, feature_group=feature_group,
        apply_embargo=apply_embargo, verbose=False,
    )

    if model_name in T.SKLEARN_MODELS:
        Xtr, ytr = T.loader_to_numpy(train_loader)
        Xte, y_true = T.loader_to_numpy(test_loader)
        Xfit, yfit = T.flatten(Xtr), ytr   # training partition only, as in train.py
        use_seeds = [seeds[0]] if model_name in T.DETERMINISTIC_MODELS else seeds
        rows = []
        for s in use_seeds:
            clf = T.build_sklearn_model(model_name, s)
            if model_name in T.SAMPLE_WEIGHTED_MODELS:
                from sklearn.utils.class_weight import compute_sample_weight
                clf.fit(Xfit, yfit, sample_weight=compute_sample_weight("balanced", yfit))
            else:
                clf.fit(Xfit, yfit)
            rows.append(T.compute_metrics(y_true, clf.predict(T.flatten(Xte))))
    else:
        Xb, _ = next(iter(train_loader))
        input_size = int(Xb.shape[-1])
        rows, y_true = [], None
        for s in seeds:
            y_true, proba, _ = T.train_torch_seed(
                model_name, train_loader, val_loader, test_loader, device, input_size, s,
                os.path.join("/tmp", f"rob_{model_name}_{feature_group}_{s}.pth"),
            )
            rows.append(T.compute_metrics(y_true, proba.argmax(1)))

    counts = np.bincount(np.asarray(y_true), minlength=3)
    out = {"n_test": int(len(y_true)),
           "pct_up": round(100 * counts[0] / len(y_true), 1),
           "pct_down": round(100 * counts[1] / len(y_true), 1),
           "pct_neutral": round(100 * counts[2] / len(y_true), 1)}
    for k in ("accuracy", "macro_f1", "balanced_accuracy", "mcc", "custom_cost"):
        v = [r[k] for r in rows]
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))

    abstain = np.full(len(y_true), 2)
    out["abstention_cost"] = T.custom_cost(np.asarray(y_true), abstain)
    out["abstention_accuracy"] = float((np.asarray(y_true) == 2).mean())
    return out


# ---------------------------------------------------------------------------


@contextmanager
def label_settings():
    """Save and restore the label globals in dataloaders, including on error."""
    saved = (DL.PRICE_CHANGE_THRESHOLD, DL.VOLATILITY_SCALED_LABEL,
             DL.VOLATILITY_MULTIPLIER, DL.EMBARGO_DAYS)
    try:
        yield
    finally:
        (DL.PRICE_CHANGE_THRESHOLD, DL.VOLATILITY_SCALED_LABEL,
         DL.VOLATILITY_MULTIPLIER, DL.EMBARGO_DAYS) = saved


def block_a_thresholds(device) -> pd.DataFrame:
    print("\n=== A. LABEL THRESHOLD SENSITIVITY ===")
    rows = []

    for thr in THRESHOLD_GRID:
        DL.VOLATILITY_SCALED_LABEL = False
        DL.PRICE_CHANGE_THRESHOLD = thr
        for m in MODELS:
            r = evaluate(m, "all", device)
            r.update({"label": f"fixed +/-{thr:.0%}", "model": m})
            rows.append(r)
            print(f"  fixed {thr:.0%} | {m:10s} n={r['n_test']} neutral={r['pct_neutral']}% "
                  f"macro-F1={r['macro_f1_mean']:.3f} cost={r['custom_cost_mean']:.3f} "
                  f"(abstain {r['abstention_cost']:.3f})")

    for mult in VOL_MULTIPLIERS:
        DL.VOLATILITY_SCALED_LABEL = True
        DL.VOLATILITY_MULTIPLIER = mult
        for m in MODELS:
            r = evaluate(m, "all", device)
            r.update({"label": f"vol-scaled x{mult}", "model": m})
            rows.append(r)
            print(f"  vol x{mult} | {m:10s} n={r['n_test']} neutral={r['pct_neutral']}% "
                  f"macro-F1={r['macro_f1_mean']:.3f} cost={r['custom_cost_mean']:.3f} "
                  f"(abstain {r['abstention_cost']:.3f})")

    return pd.DataFrame(rows)


def block_b_modalities(device) -> pd.DataFrame:
    print("\n=== B. SINGLE-MODALITY ABLATIONS ===")
    rows = []
    for g in MODALITIES:
        for m in MODELS:
            r = evaluate(m, g, device)
            r.update({"feature_group": g, "model": m})
            rows.append(r)
            print(f"  {g:18s} | {m:10s} macro-F1={r['macro_f1_mean']:.3f}"
                  f"+/-{r['macro_f1_std']:.3f} cost={r['custom_cost_mean']:.3f}")
    return pd.DataFrame(rows)


MIN_TEST_EVENTS = 40   # below this the test partition is not worth scoring


def block_c_embargo(device) -> pd.DataFrame:
    print("\n=== C. EMBARGO LENGTH SWEEP ===")
    rows = []

    for days in (0, 7, 14, 21, 30):
        DL.EMBARGO_DAYS = days
        loader, _ = create_dataloader(batch_size=64, split="test", feature_group="all",
                                      apply_embargo=True, verbose=False)
        n_test = len(loader.dataset)
        print(f"  embargo={days:2d}d -> test events = {n_test}")

        if n_test < MIN_TEST_EVENTS:
            rows.append({"embargo_days": days, "n_test": n_test, "model": None,
                         "note": "test partition too small to evaluate"})
            continue

        for m in MODELS:
            r = evaluate(m, "all", device, apply_embargo=True)
            r.update({"embargo_days": days, "model": m, "note": ""})
            rows.append(r)
            print(f"    {m:10s} macro-F1={r['macro_f1_mean']:.3f} "
                  f"cost={r['custom_cost_mean']:.3f} (abstain {r['abstention_cost']:.3f})")

    return pd.DataFrame(rows)


def main():
    device = T.setup_device()
    print(f"Robustness checks | seeds={SEEDS} | device={device}")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    with label_settings():
        a = block_a_thresholds(device)
    a.to_csv(os.path.join(FIGURES_DIR, "robustness_thresholds.csv"), index=False)

    with label_settings():
        b = block_b_modalities(device)
    b.to_csv(os.path.join(FIGURES_DIR, "robustness_modalities.csv"), index=False)

    with label_settings():
        c = block_c_embargo(device)
    c.to_csv(os.path.join(FIGURES_DIR, "robustness_embargo.csv"), index=False)

    print("\nWrote: robustness_thresholds.csv, robustness_modalities.csv, robustness_embargo.csv")


if __name__ == "__main__":
    main()
