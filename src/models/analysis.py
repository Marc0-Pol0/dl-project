"""
Post-hoc analysis. Reads the predicted probabilities written by train.py and
produces everything that does not require retraining:

  1. Bootstrap confidence intervals for every headline metric

Two sources of uncertainty are reported side by side: the spread across seeds
(mean +/- sd, as in Table 1) and the bootstrap interval across test
observations. Paired tests use the seed-averaged (ensemble) predictions, which
differ from the per-seed mean.
  2. Paired significance tests between models (McNemar; permutation on cost)
  3. One-versus-rest ROC-AUC and precision-recall AUC
  4. Brier scores and calibration curves
  5. Confidence-threshold sweep: the coverage-versus-risk frontier, which lets
     models with different implicit abstention behaviour be compared on common
     ground instead of as two fixed classifiers
  6. Sensitivity of the ranking to the UP<->DOWN cost ratio, including the point
     at which each model overtakes abstention

Run after train.py:  python src/models/analysis.py
"""

import os
import json
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRED_DIR = "./src/figures/predictions"
FIGURES_DIR = "./src/figures"
N_BOOTSTRAP = 5000
RNG = np.random.default_rng(0)

COST_RATIO = 3.0  # must match Config.COST_RATIO_UP_DOWN in train.py

CLASS_NAMES = ["UP", "DOWN", "NEUTRAL"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def custom_cost(y_true, y_pred, ratio=COST_RATIO):
    wrong = y_true != y_pred
    swap = ((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))
    return float((wrong.sum() + (ratio - 1.0) * swap.sum()) / len(y_true))


def per_observation_cost(y_true, y_pred, ratio=COST_RATIO):
    """Cost of each individual prediction, needed for the paired permutation test."""
    c = np.zeros(len(y_true), dtype=float)
    wrong = y_true != y_pred
    swap = ((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))
    c[wrong] = 1.0
    c[swap] = ratio
    return c


METRICS = {
    "accuracy": lambda yt, yp: float((yt == yp).mean()),
    "macro_f1": lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0)),
    "balanced_accuracy": lambda yt, yp: float(balanced_accuracy_score(yt, yp)),
    "mcc": lambda yt, yp: float(matthews_corrcoef(yt, yp)),
    "custom_cost": lambda yt, yp: custom_cost(yt, yp),
}


# ---------------------------------------------------------------------------
# 1. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap over test observations."""
    idx = np.arange(len(y_true))
    vals = np.empty(n)
    for i in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        vals[i] = metric_fn(y_true[b], y_pred[b])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# 2. Paired significance tests
# ---------------------------------------------------------------------------

def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test on overall correctness (discordant pairs)."""
    a_ok, b_ok = (pred_a == y_true), (pred_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return n01, n10, p


def paired_permutation(y_true, pred_a, pred_b, ratio=COST_RATIO, n=10000):
    """Paired permutation test on the per-observation cost difference."""
    ca = per_observation_cost(y_true, pred_a, ratio)
    cb = per_observation_cost(y_true, pred_b, ratio)
    d = ca - cb
    obs = float(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


# ---------------------------------------------------------------------------
# 3-4. Ranking, calibration
# ---------------------------------------------------------------------------

def ranking_metrics(y_true, proba):
    """One-versus-rest ROC-AUC and PR-AUC, plus per-class Brier scores.

    Both ROC-AUC and PR-AUC are reported; PR-AUC is more informative under
    class imbalance.
    """
    out = {}
    for cls, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            out[f"roc_auc_{name}"] = float("nan")
            out[f"pr_auc_{name}"] = float("nan")
            out[f"brier_{name}"] = float("nan")
            continue
        out[f"roc_auc_{name}"] = float(roc_auc_score(y_bin, proba[:, cls]))
        out[f"pr_auc_{name}"] = float(average_precision_score(y_bin, proba[:, cls]))
        out[f"brier_{name}"] = float(brier_score_loss(y_bin, proba[:, cls]))
    finite = [v for k, v in out.items() if k.startswith("roc_auc") and np.isfinite(v)]
    out["roc_auc_macro"] = float(np.mean(finite)) if finite else float("nan")
    out["brier_mean"] = float(np.nanmean([out[f"brier_{n}"] for n in CLASS_NAMES]))
    return out


def calibration_curve_data(y_true, proba, cls, n_bins=8):
    y_bin = (y_true == cls).astype(int)
    p = proba[:, cls]
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            xs.append(float(p[m].mean()))
            ys.append(float(y_bin[m].mean()))
            ns.append(int(m.sum()))
    return xs, ys, ns


# ---------------------------------------------------------------------------
# 5. Coverage-versus-risk frontier
# ---------------------------------------------------------------------------

def coverage_risk_frontier(y_true, proba, thresholds=None, ratio=COST_RATIO):
    """
    Issue a directional signal only when max(P(UP), P(DOWN)) exceeds the
    threshold, else abstain. Sweeping the threshold traces the coverage/cost
    trade-off on a common axis across models.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 25)

    rows = []
    for t in thresholds:
        directional_conf = proba[:, [0, 1]].max(axis=1)
        directional_pick = proba[:, [0, 1]].argmax(axis=1)
        y_pred = np.where(directional_conf >= t, directional_pick, 2)

        acted = y_pred != 2
        n_acted = int(acted.sum())
        hit = float((y_pred[acted] == y_true[acted]).mean()) if n_acted else float("nan")
        swap = int(np.sum(((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))))

        rows.append({
            "threshold": float(t),
            "coverage": n_acted / len(y_true),
            "n_signals": n_acted,
            "directional_hit_rate": hit,
            "n_up_down_swaps": swap,
            "custom_cost": custom_cost(y_true, y_pred, ratio),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Cost-ratio sensitivity
# ---------------------------------------------------------------------------

def cost_ratio_sweep(y_true, preds_by_model, ratios=(1.0, 1.5, 2.0, 3.0, 5.0, 10.0)):
    """
    Cost under a range of UP<->DOWN penalty ratios, alongside abstention.
    """
    rows = []
    abstain = np.full(len(y_true), 2)
    for r in ratios:
        row = {"cost_ratio": r, "always_neutral": custom_cost(y_true, abstain, r)}
        for tag, yp in preds_by_model.items():
            row[tag] = custom_cost(y_true, yp, r)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_predictions():
    if not os.path.isdir(PRED_DIR):
        raise SystemExit(f"{PRED_DIR} not found. Run train.py first.")
    out = {}
    for fn in sorted(os.listdir(PRED_DIR)):
        if fn.endswith(".npz"):
            d = np.load(os.path.join(PRED_DIR, fn))
            out[fn[:-4]] = {"y_true": d["y_true"].astype(int), "proba": d["proba"]}
    if not out:
        raise SystemExit(f"No .npz files in {PRED_DIR}. Run train.py first.")
    return out


def main():
    preds = load_predictions()
    y_true = next(iter(preds.values()))["y_true"]
    n = len(y_true)

    # All configurations must share one test partition for the paired tests.
    for tag, d in preds.items():
        if not np.array_equal(d["y_true"], y_true):
            raise SystemExit(
                f"Configuration {tag} has a different test partition. "
                "Re-run train.py so that all configurations share one test set."
            )
    print(f"Loaded {len(preds)} configurations | test n = {n}")
    print(f"Test class counts: UP={int((y_true==0).sum())} "
          f"DOWN={int((y_true==1).sum())} NEUTRAL={int((y_true==2).sum())}\n")

    # Seed-averaged probabilities -> the ensemble used for the paired tests.
    mean_proba = {t: d["proba"].mean(axis=0) for t, d in preds.items()}
    point_pred = {t: p.argmax(1) for t, p in mean_proba.items()}
    point_pred["always_neutral"] = np.full(n, 2)

    # Per-seed metrics, so the seed spread reported in Table 1 can be shown next
    # to the bootstrap interval of the ensemble.
    per_seed_stats = {}
    for tag, d in preds.items():
        vals = {m: [] for m in METRICS}
        for s_i in range(d["proba"].shape[0]):
            yp_s = d["proba"][s_i].argmax(1)
            for m, fn in METRICS.items():
                vals[m].append(fn(y_true, yp_s))
        per_seed_stats[tag] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in vals.items()}

    # --- 1. Uncertainty -----------------------------------------------------
    print(f"=== UNCERTAINTY: seed spread vs bootstrap over test observations ===")
    print(f"    'seeds'     = mean +/- sd over independent runs (matches Table 1)")
    print(f"    'ensemble'  = seed-averaged probabilities, with {N_BOOTSTRAP}-resample 95% CI")
    ci_rows = []
    for tag, yp in point_pred.items():
        row = {"model": tag}
        for mname, fn in METRICS.items():
            v, lo, hi = bootstrap_ci(y_true, yp, fn)
            if tag in per_seed_stats:
                mu, sd = per_seed_stats[tag][mname]
                row[f"{mname}_seeds"] = f"{mu:.3f}+/-{sd:.3f}"
            else:
                row[f"{mname}_seeds"] = "-"
            row[f"{mname}_ensemble"] = f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    for mname in METRICS:
        print(f"\n-- {mname} --")
        print(ci_df[["model", f"{mname}_seeds", f"{mname}_ensemble"]].to_string(index=False))
    ci_df.to_csv(os.path.join(FIGURES_DIR, "bootstrap_ci.csv"), index=False)

    # --- 2. Paired tests ----------------------------------------------------
    print("\n=== PAIRED TESTS vs ALWAYS-NEUTRAL (on seed-ensemble predictions) ===")
    test_rows = []
    for tag, yp in point_pred.items():
        if tag == "always_neutral":
            continue
        n01, n10, p_mc = mcnemar_exact(y_true, yp, point_pred["always_neutral"])
        d_cost, p_perm = paired_permutation(y_true, yp, point_pred["always_neutral"])
        test_rows.append({"model": tag, "mcnemar_n01": n01, "mcnemar_n10": n10,
                          "mcnemar_p": round(p_mc, 4),
                          "cost_diff_vs_abstain": round(d_cost, 4),
                          "permutation_p": round(p_perm, 4)})
    t_df = pd.DataFrame(test_rows)
    print(t_df.to_string(index=False))
    t_df.to_csv(os.path.join(FIGURES_DIR, "paired_tests_vs_abstention.csv"), index=False)

    print("\n=== PAIRED TESTS BETWEEN MODELS (on seed-ensemble predictions) ===")
    pair_rows = []
    tags = [t for t in point_pred if t != "always_neutral"]
    for a, b in itertools.combinations(tags, 2):
        n01, n10, p_mc = mcnemar_exact(y_true, point_pred[a], point_pred[b])
        d_cost, p_perm = paired_permutation(y_true, point_pred[a], point_pred[b])
        pair_rows.append({"model_a": a, "model_b": b, "mcnemar_p": round(p_mc, 4),
                          "cost_diff": round(d_cost, 4), "permutation_p": round(p_perm, 4)})
    p_df = pd.DataFrame(pair_rows)
    print(p_df.to_string(index=False))
    p_df.to_csv(os.path.join(FIGURES_DIR, "paired_tests_between_models.csv"), index=False)

    # --- 3-4. Ranking metrics and calibration -------------------------------
    print("\n=== RANKING METRICS AND CALIBRATION (seed-averaged probabilities) ===")
    rank_rows = []
    for tag, proba in mean_proba.items():
        r = ranking_metrics(y_true, proba)
        r["model"] = tag
        rank_rows.append(r)
    r_df = pd.DataFrame(rank_rows).set_index("model")
    cols = ["roc_auc_macro"] + [f"pr_auc_{c}" for c in CLASS_NAMES] + ["brier_mean"]
    print(r_df[cols].round(3).to_string())
    r_df.round(6).to_csv(os.path.join(FIGURES_DIR, "ranking_calibration.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for cls, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for tag, proba in mean_proba.items():
            xs, ys, _ = calibration_curve_data(y_true, proba, cls)
            ax.plot(xs, ys, marker="o", ms=3, lw=1, label=tag)
        ax.set_title(f"Calibration: {CLASS_NAMES[cls]}")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- 5. Coverage-risk frontier -----------------------------------------
    print("\n=== COVERAGE-VERSUS-RISK FRONTIER (excerpt) ===")
    frontier_all = []
    plt.figure(figsize=(7, 5))
    for tag, proba in mean_proba.items():
        f = coverage_risk_frontier(y_true, proba)
        f["model"] = tag
        frontier_all.append(f)
        plt.plot(f["coverage"], f["custom_cost"], marker=".", lw=1, label=tag)
    plt.axhline(custom_cost(y_true, np.full(n, 2)), color="k", ls="--", lw=1,
                label="always-NEUTRAL")
    plt.xlabel("coverage (fraction of events with a directional signal)")
    plt.ylabel("average custom cost")
    plt.title("Coverage versus risk")
    plt.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "coverage_risk_frontier.png"), dpi=200, bbox_inches="tight")
    plt.close()

    fr = pd.concat(frontier_all, ignore_index=True)
    fr.to_csv(os.path.join(FIGURES_DIR, "coverage_risk_frontier.csv"), index=False)
    print(fr[fr["threshold"].round(2).isin([0.35, 0.50, 0.65])]
          [["model", "threshold", "coverage", "directional_hit_rate", "custom_cost"]]
          .round(3).to_string(index=False))

    # --- 6. Cost-ratio sweep ------------------------------------------------
    print("\n=== COST-RATIO SENSITIVITY ===")
    sweep = cost_ratio_sweep(y_true, {t: p for t, p in point_pred.items() if t != "always_neutral"})
    print(sweep.round(3).to_string(index=False))
    sweep.to_csv(os.path.join(FIGURES_DIR, "cost_ratio_sweep.csv"), index=False)

    print("\nWrote: bootstrap_ci.csv, paired_tests_vs_abstention.csv,")
    print("       paired_tests_between_models.csv, ranking_calibration.csv,")
    print("       coverage_risk_frontier.{csv,png}, cost_ratio_sweep.csv,")
    print("       calibration_curves.png")


if __name__ == "__main__":
    main()
