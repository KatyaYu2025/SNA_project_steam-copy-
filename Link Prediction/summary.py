"""
Combine all link prediction results into a single comparison table.
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path("Link Prediction")

RESULT_FILES = {
    "Heuristic — Jaccard": (OUT_DIR / "heuristic_results.json", "jaccard"),
    "Heuristic — Weighted Jaccard": (OUT_DIR / "heuristic_results.json", "weighted_jaccard"),
    "Heuristic — Common Neighbors": (OUT_DIR / "heuristic_results.json", "common_neighbors"),
    "Heuristic — Cosine": (OUT_DIR / "heuristic_results.json", "cosine"),
    "Heuristic — Adamic-Adar": (OUT_DIR / "heuristic_results.json", "adamic_adar"),
    "Heuristic — Pref. Attachment": (OUT_DIR / "heuristic_results.json", "preferential_attachment"),
    "Heuristic — Playtime Correlation": (OUT_DIR / "heuristic_results.json", "playtime_correlation"),
}

ML_FILE = OUT_DIR / "ml_results.json"
N2V_FILE = OUT_DIR / "node2vec_results.json"
GAE_FILE = OUT_DIR / "gae_results.json"

SUMMARY_PATH = OUT_DIR / "link_prediction_summary.json"


def load_results():
    all_results = {}

    # Heuristic baselines
    if HEUR_RES.exists():
        with open(HEUR_RES) as f:
            heur = json.load(f)
        for label, (fpath, key) in RESULT_FILES.items():
            if key in heur:
                all_results[label] = heur[key]
    else:
        log.warning("heuristic_results.json not found")

    # ML classifiers
    if ML_FILE.exists():
        with open(ML_FILE) as f:
            ml = json.load(f)
        for name, res in ml.items():
            all_results[f"ML — {name}"] = res["mean"]

    # Node2Vec
    if N2V_FILE.exists():
        with open(N2V_FILE) as f:
            n2v = json.load(f)
        all_results["Node2Vec"] = n2v

    # GAE
    if GAE_FILE.exists():
        with open(GAE_FILE) as f:
            gae = json.load(f)
        all_results["GAE (Graph Autoencoder)"] = gae

    return all_results


def main():
    global HEUR_RES
    HEUR_RES = RESULT_FILES["Heuristic — Jaccard"][0]

    all_results = load_results()
    if not all_results:
        log.error("No results found. Run heuristic_baseline.py, ml_classifier.py, node2vec.py, and gae.py first.")
        sys.exit(1)

    # Build summary table
    header = ["Method", "AUC-ROC", "AUC-PR", "P@10", "P@50", "P@100", "R@50"]
    rows = []
    for method, res in all_results.items():
        auc = res.get("auc_roc", "—")
        aupr = res.get("auc_pr", "—")
        p10 = res.get("precision@10", "—")
        p50 = res.get("precision@50", "—")
        p100 = res.get("precision@100", "—")
        r50 = res.get("recall@50", "—")
        rows.append((method, auc, aupr, p10, p50, p100, r50))

    col_widths = [
        max(len(r[i]) if isinstance(r[i], str) else len(f"{r[i]:.4f}") for r in rows + [tuple(header)])
        for i in range(len(header))
    ]
    # Adjust method column
    col_widths[0] = max(len(r[0]) for r in rows + [tuple(header)])

    def fmt(val):
        if isinstance(val, str):
            return val.ljust(col_widths[header.index("Method") if val == r[0] else ...])
        return f"{val:.4f}".ljust(col_widths[1])

    # Better to just print table
    sep = "=" * (sum(col_widths) + 3 * len(header))
    log.info("")
    log.info("=" * (sum(col_widths) + 3 * len(header)))
    header_str = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    log.info(header_str)
    log.info("=" * (sum(col_widths) + 3 * len(header)))
    for row in rows:
        vals = []
        for i, v in enumerate(row):
            if isinstance(v, str):
                vals.append(v.ljust(col_widths[i]))
            elif v == "—" or v is None:
                vals.append("—".ljust(col_widths[i]))
            else:
                vals.append(f"{v:.4f}".ljust(col_widths[i]))
        log.info(" | ".join(vals))
    log.info("=" * (sum(col_widths) + 3 * len(header)))

    # Save combined summary
    summary = {}
    for method, res in all_results.items():
        summary[method] = {
            k: v for k, v in res.items()
            if k in ("auc_roc", "auc_pr")
            or k.startswith("precision@")
            or k.startswith("recall@")
        }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary to %s", SUMMARY_PATH)

    # Identify best model
    best_auc = max(
        ((m, r["auc_roc"]) for m, r in all_results.items() if isinstance(r.get("auc_roc"), (int, float))),
        key=lambda x: x[1],
    )
    log.info("")
    log.info("Best model by AUC-ROC: %s (%.4f)", best_auc[0], best_auc[1])


if __name__ == "__main__":
    main()
