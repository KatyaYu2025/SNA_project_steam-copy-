"""
Heuristic Baselines for Link Prediction.

Evaluates each feature as a standalone threshold-based predictor:
  Jaccard, Weighted Jaccard, Common Neighbors, Cosine, Adamic-Adar,
  Preferential Attachment, Playtime Correlation.

Outputs:
  Link Prediction/heuristic_results.json — AUC-ROC, Precision@k, Recall@k
  Link Prediction/auc_comparison.png — bar chart of AUC for each feature
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_PATH = "graphs/player_features.csv"
OUT_DIR = Path("Link Prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "heuristic_results.json"
AUC_PLOT_PATH = OUT_DIR / "auc_comparison.png"

# Features to evaluate as stand-alone predictors
FEATURES = [
    "jaccard",
    "weighted_jaccard",
    "common_neighbors",
    "cosine",
    "adamic_adar",
    "preferential_attachment",
    "playtime_correlation",
]

K_VALUES = [10, 20, 50, 100, 200]


def precision_at_k(df, score_col, k):
    top = df.nlargest(k, score_col)
    return top["is_friend"].mean()


def recall_at_k(df, score_col, k, total_pos):
    top = df.nlargest(k, score_col)
    return top["is_friend"].sum() / total_pos if total_pos > 0 else 0.0


def evaluate_one(df, feature):
    vals = df[feature].fillna(0)
    y = df["is_friend"].astype(int)
    total_pos = y.sum()

    auc_val = roc_auc_score(y, vals)
    precisions, recalls, thresholds = precision_recall_curve(y, vals)
    pr_auc = auc(recalls, precisions)

    results = {
        "auc_roc": round(auc_val, 4),
        "auc_pr": round(pr_auc, 4),
    }
    for k in K_VALUES:
        results[f"precision@{k}"] = round(precision_at_k(df, feature, k), 4)
        results[f"recall@{k}"] = round(recall_at_k(df, feature, k, total_pos), 4)

    return results


def main():
    log.info("Loading features from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %d rows", len(df))

    results = {}
    for feat in FEATURES:
        results[feat] = evaluate_one(df, feat)
        log.info(
            "  %-24s AUC=%-8s  P@50=%-8s  R@50=%-8s",
            feat,
            results[feat]["auc_roc"],
            results[feat]["precision@50"],
            results[feat]["recall@50"],
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", RESULTS_PATH)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = []
        aucs = []
        for feat in FEATURES:
            names.append(feat.replace("_", "\n"))
            aucs.append(results[feat]["auc_roc"])

        colors = ["#2ecc71" if a >= 0.8 else ("#f39c12" if a >= 0.7 else "#e74c3c") for a in aucs]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(names)), aucs, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("AUC-ROC")
        ax.set_title("Heuristic Baseline: AUC-ROC per Feature", fontsize=13)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(AUC_PLOT_PATH, dpi=150)
        log.info("Saved AUC comparison plot to %s", AUC_PLOT_PATH)
    except Exception as e:
        log.warning("Could not generate plot: %s", e)

    # Combined: rank by Jaccard then break ties
    log.info("")
    log.info("=== Combined Jaccard threshold sweep ===")
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        pred = (df["jaccard"] >= thresh).astype(int)
        y = df["is_friend"].astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        log.info("  Jaccard>=%.2f  precision=%.4f  recall=%.4f  tp=%d  fp=%d", thresh, prec, rec, tp, fp)


if __name__ == "__main__":
    main()
