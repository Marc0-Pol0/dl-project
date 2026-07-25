"""
Main experiment grid.

Design notes, all of which are reported in the manuscript:
  * Strictly chronological train / validation / test partitions.
  * Neural checkpoint selection uses the validation partition; the test
    partition is scored once, after the checkpoint is fixed.
  * Neural configurations are repeated over Config.SEEDS (mean +/- sd).
  * Predicted probabilities are written to disk; analysis.py consumes them
    without retraining.
"""

import os
import csv
import json
import random
import time
import platform

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataloaders import create_dataloader, partition_report
from model import StockLSTM, StockTransformer


class Config:
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    OUTPUT_SIZE = 3  # UP=0, DOWN=1, NEUTRAL=2

    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4

    CLASS_WEIGHTS_TORCH = [1.0, 1.3, 1.0]

    SEEDS = [0, 1, 2, 3, 4]

    # (model, feature_group) pairs forming the main grid.
    GRID = [
        (m, g)
        for m in ("logreg", "random_forest", "gradient_boosting", "mlp", "lstm", "attention")
        for g in ("all", "no_sentiment")
    ]

    NETWORKS_DIR = "./networks"
    FIGURES_DIR = "./src/figures"
    PRED_DIR = "./src/figures/predictions"

    # Cost of an UP<->DOWN confusion relative to any other error.
    COST_RATIO_UP_DOWN = 3.0

    # See EMBARGO_DAYS in dataloaders.py. robustness.py sweeps it.
    APPLY_EMBARGO = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_device():
    """
    CUDA if present, otherwise CPU.

    Apple's MPS backend is only used from torch 2.0 onwards: in 1.x it produced
    wrong results for recurrent layers on some builds, and these models are small
    enough that CPU costs minutes, not hours. Set FORCE_MPS=1 to override.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_ok and (torch.__version__ >= "2" or os.environ.get("FORCE_MPS") == "1"):
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


MIN_VERSIONS = {
    "python": (3, 9),
    "numpy": (1, 17),      # np.random.default_rng
    "scipy": (1, 7),       # scipy.stats.binomtest, used by analysis.py
    "scikit-learn": (1, 0),  # HistGradientBoostingClassifier out of experimental
    "torch": (1, 9),       # batch_first in TransformerEncoderLayer
}


def _as_tuple(v: str):
    parts = []
    for chunk in v.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_environment() -> None:
    """Fail early on an unsupported dependency version."""
    import scipy, sklearn
    found = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
    }
    problems = [
        f"{pkg} {found[pkg]} is too old (need >= {'.'.join(map(str, req))})"
        for pkg, req in MIN_VERSIONS.items()
        if _as_tuple(found[pkg]) < req
    ]
    if problems:
        raise SystemExit(
            "Unsupported environment:\n  " + "\n  ".join(problems)
            + "\n\nUpgrade with, for example:  pip install -U scikit-learn scipy"
        )


def environment_report() -> dict:
    """Exact dependency versions, recorded for reproducibility."""
    import sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": str(setup_device()),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def custom_cost(y_true: np.ndarray, y_pred: np.ndarray, ratio: float = None) -> float:
    """Average cost: 0 if correct, `ratio` for an UP<->DOWN confusion, 1 otherwise."""
    ratio = Config.COST_RATIO_UP_DOWN if ratio is None else ratio
    wrong = y_true != y_pred
    swap = ((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))
    return float((wrong.sum() + (ratio - 1.0) * swap.sum()) / len(y_true))


def abstention_crossover_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cost ratio below which the model beats the always-NEUTRAL baseline; above it,
    abstaining is cheaper.

    When the model never confuses UP with DOWN its cost does not depend on the
    ratio, so the comparison is settled by the error counts alone: +inf if it is
    cheaper than abstention at every ratio, -inf if it is cheaper at none.
    """
    n_wrong = int((y_true != y_pred).sum())
    n_swap = int((((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))).sum())
    n_abstain_err = int((y_true != 2).sum())

    if n_swap == 0:
        # Cost does not depend on the ratio at all, so the comparison with
        # abstention is settled by the error counts alone: +inf when the model is
        # cheaper at every ratio, -inf when it is never cheaper. Returning +inf
        # unconditionally would report a model that simply makes more mistakes
        # than abstention as beating it everywhere.
        return float("inf") if n_wrong < n_abstain_err else float("-inf")

    return 1.0 + (n_abstain_err - n_wrong) / n_swap


def compute_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    out = {
        "accuracy": float(np.mean(yt == yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "custom_cost": custom_cost(yt, yp),
        "crossover_ratio": abstention_crossover_ratio(yt, yp),
    }

    # Per-class precision / recall / F1 plus prediction counts. A class that is
    # never predicted has undefined precision, reported as NaN not 0.0.
    for cls, name in enumerate(["up", "down", "neutral"]):
        n_pred = int((yp == cls).sum())
        out[f"precision_{name}"] = (
            float(precision_score(yt, yp, labels=[cls], average="macro", zero_division=0))
            if n_pred > 0 else float("nan")
        )
        out[f"recall_{name}"] = float(recall_score(yt, yp, labels=[cls], average="macro", zero_division=0))
        out[f"f1_{name}"] = float(f1_score(yt, yp, labels=[cls], average="macro", zero_division=0))
        out[f"n_predicted_{name}"] = n_pred

    return out


def naive_baselines(y_true) -> dict[str, dict]:
    """always-NEUTRAL (abstention / majority), always-UP, and class-prior random."""
    yt = np.asarray(y_true, dtype=int)
    n = len(yt)
    out = {
        "always_neutral": compute_metrics(yt, np.full(n, 2)),
        "always_up": compute_metrics(yt, np.full(n, 0)),
    }
    prior = np.bincount(yt, minlength=3) / n
    rng = np.random.default_rng(0)
    draws = [compute_metrics(yt, rng.choice(3, size=n, p=prior)) for _ in range(1000)]
    out["class_prior_random"] = {
        k: float(np.nanmean([d[k] for d in draws]))
        for k in ("accuracy", "macro_f1", "balanced_accuracy", "mcc", "custom_cost")
    }
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def loader_to_numpy(loader) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a loader's dataset in dataset order (unshuffled), so the
    design matrix does not depend on the global RNG state."""
    deterministic = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0
    )
    Xs, ys = [], []
    for Xb, yb in deterministic:
        Xs.append(Xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(int)


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1) if X.ndim == 3 else X


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_torch_model(name: str, input_size: int, device):
    if name == "attention":
        return StockTransformer(
            input_size=input_size, d_model=Config.HIDDEN_SIZE, nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS, dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE,
        ).to(device)
    if name == "lstm":
        return StockLSTM(
            input_size=input_size, hidden_size=Config.HIDDEN_SIZE, num_layers=Config.NUM_LAYERS,
            output_size=Config.OUTPUT_SIZE, dropout_rate=Config.DROPOUT_RATE,
        ).to(device)
    raise ValueError(name)


def build_sklearn_model(name: str, seed: int):
    """
    Conventional benchmarks (logreg, random forest, gradient boosting, MLP),
    at library-default hyperparameters.
    """
    if name == "logreg":
        return LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000,
            tol=1e-8,  # tightened: at the default tolerance lbfgs stops early
                       # and the test macro-F1 moves by ~0.01 between runs
            class_weight="balanced",
        )
    if name == "random_forest":
        # Library defaults, no tuning; only class balancing and the seed are set.
        return RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name == "gradient_boosting":
        # Library defaults; balanced via sample_weight at fit time (the
        # class_weight constructor argument requires scikit-learn >= 1.4).
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "mlp":
        # Library defaults; max_iter raised only so the optimiser can converge.
        return MLPClassifier(max_iter=1000, random_state=seed)
    raise ValueError(name)


SKLEARN_MODELS = ("logreg", "random_forest", "gradient_boosting", "mlp")

# Balanced via sample_weight at fit time instead of a class_weight argument.
SAMPLE_WEIGHTED_MODELS = ("gradient_boosting",)

# Seed-independent fits (run once). Logreg is convex; the HGB booster is
# deterministic at this sample size. Random forest and the MLP are seed-
# dependent and use Config.SEEDS.
DETERMINISTIC_MODELS = ("logreg", "gradient_boosting")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        loss = criterion(model(Xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            total += criterion(model(Xb), yb).item()
    return total / len(loader)


def predict_proba_torch(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(yb.numpy())
    return np.concatenate(targets).astype(int), np.concatenate(probs)


def train_torch_seed(name, train_loader, val_loader, test_loader, device, input_size, seed, save_path):
    """Train one seed, keeping the checkpoint with the lowest VALIDATION loss."""
    set_seed(seed)
    model = build_torch_model(name, input_size, device)
    weights = torch.tensor(Config.CLASS_WEIGHTS_TORCH, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(Config.NUM_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, device)
        vl = eval_loss(model, val_loader, criterion, device)
        if vl < best_val:
            best_val, best_epoch = vl, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)

    y_true, proba = predict_proba_torch(model, test_loader, device)
    return y_true, proba, {"best_val_loss": best_val, "best_epoch": best_epoch}


def train_sklearn_seed(name, Xtr, ytr, Xte, seed, save_path):
    clf = build_sklearn_model(name, seed)
    if name in SAMPLE_WEIGHTED_MODELS:
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        clf.fit(Xtr, ytr)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(clf, save_path)

    # Map predict_proba columns (ordered by clf.classes_) back to fixed 0/1/2
    # positions, in case a class is absent from the training partition.
    proba = clf.predict_proba(Xte)
    full = np.zeros((len(Xte), Config.OUTPUT_SIZE), dtype=float)
    for col, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, col]
    return full


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, tag, figures_dir, normalize=False):
    labels = [0, 1, 2]
    ticks = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    plt.figure(figsize=(7, 5.5))
    sns.set_context("paper", font_scale=1.3)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=ticks, yticklabels=ticks, cbar=True)
    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.title(f"{tag}" + (" (row-normalised)" if normalize else ""), fontweight="bold")
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"cm_{tag}{'_norm' if normalize else ''}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

AGG_KEYS = [
    "accuracy", "macro_f1", "balanced_accuracy", "mcc", "cohen_kappa", "custom_cost",
    "precision_up", "recall_up", "f1_up",
    "precision_down", "recall_down", "f1_down",
    "precision_neutral", "recall_neutral", "f1_neutral",
    "n_predicted_up", "n_predicted_down", "n_predicted_neutral",
]


def run_configuration(model_name: str, feature_group: str, device) -> dict:
    tag = f"{model_name}__{feature_group}"
    print(f"\n=== {tag} ===")

    train_loader, scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, split="train", feature_group=feature_group,
        apply_embargo=Config.APPLY_EMBARGO,
    )
    val_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE, split="val", scaler=scaler, feature_group=feature_group,
        apply_embargo=Config.APPLY_EMBARGO,
    )
    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE, split="test", scaler=scaler, feature_group=feature_group,
        apply_embargo=Config.APPLY_EMBARGO,
    )

    is_sklearn = model_name in SKLEARN_MODELS
    seeds = [Config.SEEDS[0]] if model_name in DETERMINISTIC_MODELS else Config.SEEDS

    if is_sklearn:
        Xtr, ytr = loader_to_numpy(train_loader)
        Xte, y_true = loader_to_numpy(test_loader)
        # Fitted on the training partition only (same events as the neural
        # models see for training).
        Xfit, yfit = flatten(Xtr), ytr
        Xte_f = flatten(Xte)
    else:
        Xb, _ = next(iter(train_loader))
        input_size = int(Xb.shape[-1])
        y_true = None

    per_seed, probas = [], []
    t0 = time.time()

    for seed in seeds:
        if is_sklearn:
            proba = train_sklearn_seed(
                model_name, Xfit, yfit, Xte_f, seed,
                os.path.join(Config.NETWORKS_DIR, f"{tag}_seed{seed}.joblib"),
            )
            extra = {}
        else:
            y_true, proba, extra = train_torch_seed(
                model_name, train_loader, val_loader, test_loader, device, input_size, seed,
                os.path.join(Config.NETWORKS_DIR, f"{tag}_seed{seed}.pth"),
            )

        y_pred = proba.argmax(1)
        m = compute_metrics(y_true, y_pred)
        m.update({"seed": seed, **extra})
        per_seed.append(m)
        probas.append(proba)
        print(f"  seed {seed}: acc {m['accuracy']*100:5.2f}% | macro-F1 {m['macro_f1']:.3f} | "
              f"cost {m['custom_cost']:.3f}")

    agg = {}
    for k in AGG_KEYS:
        vals = [m[k] for m in per_seed if k in m]
        agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    # Seeds can land on +inf (cheaper than abstention at every ratio) or -inf
    # (never cheaper). Averaging those with the finite ones would be meaningless,
    # so the mean covers the finite seeds only and the degenerate ones are counted.
    xs = [m["crossover_ratio"] for m in per_seed]
    finite = [x for x in xs if np.isfinite(x)]
    if finite:
        agg["crossover_ratio"] = (float(np.mean(finite)), float(np.std(finite)))
    elif xs and all(x == float("inf") for x in xs):
        agg["crossover_ratio"] = (float("inf"), 0.0)
    else:
        agg["crossover_ratio"] = (float("-inf"), 0.0)
    agg["crossover_seeds_always_better"] = (sum(x == float("inf") for x in xs), 0.0)
    agg["crossover_seeds_never_better"] = (sum(x == float("-inf") for x in xs), 0.0)

    y_pred_ref = probas[0].argmax(1)
    save_confusion_matrix(y_true, y_pred_ref, tag, Config.FIGURES_DIR)
    save_confusion_matrix(y_true, y_pred_ref, tag, Config.FIGURES_DIR, normalize=True)

    os.makedirs(Config.PRED_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(Config.PRED_DIR, f"{tag}.npz"),
        y_true=y_true, proba=np.stack(probas), seeds=np.array(seeds),
    )

    print(f"  -> acc {agg['accuracy'][0]*100:.2f}+/-{agg['accuracy'][1]*100:.2f} | "
          f"macro-F1 {agg['macro_f1'][0]:.3f}+/-{agg['macro_f1'][1]:.3f} | "
          f"cost {agg['custom_cost'][0]:.3f}+/-{agg['custom_cost'][1]:.3f} | "
          f"{time.time()-t0:.0f}s")

    return {"tag": tag, "model": model_name, "feature_group": feature_group,
            "n_seeds": len(seeds), "aggregate": agg, "per_seed": per_seed,
            "y_true": np.asarray(y_true).tolist()}


def json_safe(obj):
    """
    Replace non-finite floats (inf/-inf/NaN) with None, since these are not
    valid JSON.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def write_outputs(results, baselines, part_df, env):
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    part_path = os.path.join(Config.FIGURES_DIR, "dataset_partitions.csv")
    part_df.to_csv(part_path, index=False)

    csv_path = os.path.join(Config.FIGURES_DIR, "results_main.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "features", "n_seeds",
                    "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std",
                    "balanced_accuracy_mean", "balanced_accuracy_std",
                    "mcc_mean", "mcc_std", "custom_cost_mean", "custom_cost_std",
                    "crossover_ratio_mean",
                    "n_seeds_always_beat_abstention", "n_seeds_never_beat_abstention"])
        for name, m in baselines.items():
            w.writerow([name, "-", "-",
                        f"{m['accuracy']:.6f}", "", f"{m['macro_f1']:.6f}", "",
                        f"{m['balanced_accuracy']:.6f}", "", f"{m['mcc']:.6f}", "",
                        f"{m['custom_cost']:.6f}", "", "", "", ""])
        for r in results:
            a = r["aggregate"]
            w.writerow([r["model"], r["feature_group"], r["n_seeds"],
                        f"{a['accuracy'][0]:.6f}", f"{a['accuracy'][1]:.6f}",
                        f"{a['macro_f1'][0]:.6f}", f"{a['macro_f1'][1]:.6f}",
                        f"{a['balanced_accuracy'][0]:.6f}", f"{a['balanced_accuracy'][1]:.6f}",
                        f"{a['mcc'][0]:.6f}", f"{a['mcc'][1]:.6f}",
                        f"{a['custom_cost'][0]:.6f}", f"{a['custom_cost'][1]:.6f}",
                        f"{a['crossover_ratio'][0]:.3f}",
                        a['crossover_seeds_always_better'][0],
                        a['crossover_seeds_never_better'][0]])

    with open(os.path.join(Config.FIGURES_DIR, "results_main.json"), "w", encoding="utf-8") as f:
        payload = json_safe({
            "environment": env, "baselines": baselines, "models": results,
            "partitions": part_df.to_dict(orient="records"),
            "partition_overlap": dict(part_df.attrs),
            "config": {"seeds": Config.SEEDS, "epochs": Config.NUM_EPOCHS,
                       "lr": Config.LEARNING_RATE, "batch_size": Config.BATCH_SIZE,
                       "class_weights": Config.CLASS_WEIGHTS_TORCH,
                       "cost_ratio_up_down": Config.COST_RATIO_UP_DOWN,
                       "embargo": Config.APPLY_EMBARGO},
        })
        json.dump(payload, f, indent=1, default=str, allow_nan=False)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(Config.FIGURES_DIR, 'results_main.json')}")
    print(f"Wrote {part_path}")
    print(f"Predicted probabilities in {Config.PRED_DIR}/  (input to analysis.py)")


def main():
    check_environment()
    device = setup_device()
    env = environment_report()
    print("Environment:", json.dumps(env))
    print(f"Seeds: {Config.SEEDS} | embargo: {Config.APPLY_EMBARGO}\n")

    print("=== DATASET PARTITIONS ===")
    part_df = partition_report(apply_embargo=Config.APPLY_EMBARGO)
    print(part_df.to_string(index=False))
    print(f"firms in both train and test: {part_df.attrs['firms_train_and_test']}")
    print(f"firms in all three partitions: {part_df.attrs['firms_in_all_three']}")

    results = [run_configuration(m, g, device) for m, g in Config.GRID]

    baselines = naive_baselines(results[0]["y_true"])
    print("\n=== NAIVE BASELINES (same test partition) ===")
    for name, m in baselines.items():
        print(f"  {name:20s} acc {m['accuracy']*100:5.2f}% | macro-F1 {m['macro_f1']:.3f} | "
              f"cost {m['custom_cost']:.3f}")

    write_outputs(results, baselines, part_df, env)


if __name__ == "__main__":
    main()
