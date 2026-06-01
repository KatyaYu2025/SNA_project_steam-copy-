"""
ML Classifier for Link Prediction.

Trains Logistic Regression, Random Forest, and XGBoost classifiers
using all 8 features and evaluates with cross-validation.

Outputs:
  Link Prediction/ml_results.json        — AUC, Precision@k, feature importances
  Link Prediction/roc_comparison.png     — ROC curves for all models
  Link Prediction/pr_comparison.png      — PR curves
  Link Prediction/feature_importance.png — top features per model
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
RESULTS_PATH = OUT_DIR / "ml_results.json"
ROC_PLOT_PATH = OUT_DIR / "roc_comparison.png"
PR_PLOT_PATH = OUT_DIR / "pr_comparison.png"
FI_PLOT_PATH = OUT_DIR / "feature_importance.png"

FEATURE_COLS = [
    "jaccard",
    "weighted_jaccard",
    "common_neighbors",
    "cosine",
    "adamic_adar",
    "preferential_attachment",
    "playtime_correlation",
]

K_VALUES = [10, 20, 50, 100, 200]
N_SPLITS = 5
RANDOM_STATE = 42


def precision_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[idx].mean()


def recall_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    return y_true.iloc[idx].sum() / max(y_true.sum(), 1)


def evaluate_model(name, clf, X, y, df_orig, feature_cols):
    from sklearn.metrics import (
        accuracy_score,
        auc,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    total_pos = y.sum()

    fold_metrics = []
    all_y_true = []
    all_y_score = []
    importances_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        df_test = df_orig.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_score.extend(y_score.tolist())

        auc_val = roc_auc_score(y_test, y_score)
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(rec, prec)

        metrics = {
            "fold": fold,
            "auc_roc": round(auc_val, 4),
            "auc_pr": round(pr_auc, 4),
        }
        for k in K_VALUES:
            metrics[f"precision@{k}"] = round(
                precision_at_k(y_test.reset_index(drop=True), y_score, k), 4
            )
            metrics[f"recall@{k}"] = round(
                recall_at_k(y_test.reset_index(drop=True), y_score, k), 4
            )
        fold_metrics.append(metrics)

        if hasattr(clf, "feature_importances_"):
            importances_list.append(clf.feature_importances_)

    # Aggregate
    avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0] if k != "fold"}
    std = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0] if k != "fold"}
    results = {
        "model": name,
        "mean": {k: round(v, 4) for k, v in avg.items()},
        "std": {k: round(v, 4) for k, v in std.items()},
        "fold_results": fold_metrics,
        "total_positive_samples": int(total_pos),
    }

    if importances_list:
        results["feature_importance"] = {
            feat: round(float(np.mean(importances_list, axis=0)[i]), 4)
            for i, feat in enumerate(FEATURE_COLS)
        }

    return results, np.array(all_y_true), np.array(all_y_score)


def main():
    log.info("Loading features from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Loaded %d rows", len(df))

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_friend"].astype(int)

    log.info("Class distribution: pos=%d / neg=%d (ratio=1:%.0f)", y.sum(), (1 - y).sum(), (1 - y).sum() / max(y.sum(), 1))

    # Log-transform preferential_attachment to avoid scale issues
    X["preferential_attachment"] = np.log1p(X["preferential_attachment"])

    log.info("Scaling features...")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            scale_pos_weight=(1 - y).sum() / max(y.sum(), 1),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    all_results = {}
    roc_data = {}
    pr_data = {}

    for name, clf in models.items():
        log.info("Training %s ...", name)
        results, y_true, y_score = evaluate_model(name, clf, X_scaled, y, df, FEATURE_COLS)
        all_results[name] = results
        roc_data[name] = (y_true, y_score)
        log.info(
            "  AUC=%.4f ± %.4f  P@50=%.4f  R@50=%.4f",
            results["mean"]["auc_roc"],
            results["std"]["auc_roc"],
            results["mean"]["precision@50"],
            results["mean"]["recall@50"],
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved results to %s", RESULTS_PATH)

    _plot_roc(roc_data)
    _plot_pr(pr_data)
    _plot_feature_importance(all_results)

    log.info("")
    log.info("=== Best Model Summary ===")
    best = max(all_results.items(), key=lambda x: x[1]["mean"]["auc_roc"])
    log.info("  Best model: %s (AUC=%.4f ± %.4f)", best[0], best[1]["mean"]["auc_roc"], best[1]["std"]["auc_roc"])

    # Summary table
    log.info("")
    log.info("%-22s %-10s %-10s %-10s %-10s", "Model", "AUC-ROC", "P@10", "P@50", "P@100")
    log.info("-" * 62)
    for name, res in all_results.items():
        log.info(
            "%-22s %-10.4f %-10.4f %-10.4f %-10.4f",
            name,
            res["mean"]["auc_roc"],
            res["mean"]["precision@10"],
            res["mean"]["precision@50"],
            res["mean"]["precision@100"],
        )


def _plot_roc(roc_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (y_true, y_score) in roc_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auroc:.4f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Link Prediction Classifiers")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(ROC_PLOT_PATH, dpi=150)
        log.info("Saved ROC plot to %s", ROC_PLOT_PATH)
    except Exception as e:
        log.warning("Could not generate ROC plot: %s", e)


def _plot_pr(pr_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, auc

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (y_true, y_score) in pr_data.items():
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, label=f"{name} (AUC-PR={pr_auc:.4f})", linewidth=1.5)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves — Link Prediction Classifiers")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(PR_PLOT_PATH, dpi=150)
        log.info("Saved PR plot to %s", PR_PLOT_PATH)
    except Exception as e:
        log.warning("Could not generate PR plot: %s", e)


def _plot_feature_importance(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models_with_fi = {
            name: res.get("feature_importance")
            for name, res in all_results.items()
            if res.get("feature_importance")
        }
        if not models_with_fi:
            log.info("No feature importance data to plot")
            return

        fig, axes = plt.subplots(1, len(models_with_fi), figsize=(6 * len(models_with_fi), 5))
        if len(models_with_fi) == 1:
            axes = [axes]

        for ax, (name, fi) in zip(axes, models_with_fi.items()):
            feats = sorted(fi, key=fi.get)
            vals = [fi[f] for f in feats]
            ax.barh(range(len(feats)), vals, color="steelblue")
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=9)
            ax.set_xlabel("Importance")
            ax.set_title(name, fontsize=11)

        plt.suptitle("Feature Importance by Model", fontsize=13)
        plt.tight_layout()
        plt.savefig(FI_PLOT_PATH, dpi=150)
        log.info("Saved feature importance plot to %s", FI_PLOT_PATH)
    except Exception as e:
        log.warning("Could not generate feature importance plot: %s", e)


if __name__ == "__main__":
    main()
