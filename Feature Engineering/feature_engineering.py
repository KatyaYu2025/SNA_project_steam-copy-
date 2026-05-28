"""
Feature Engineering for Player-Player Link Prediction.

Computes all features from the project spec for each pair of players (i, j):
  Structural:  Jaccard, Weighted Jaccard (playtime), Common Neighbors, Cosine
  Graph-based: Adamic-Adar, Preferential Attachment
  Behavioral:  Playtime Correlation, Genre Overlap (stub)

Output:
  graphs/player_features.csv — one row per player pair with all features + is_friend
  graphs/feature_stats.json  — summary statistics per feature
  graphs/feature_correlation.png — feature correlation heatmap

Usage: python feature_engineering.py
"""

import json
import logging
import math
import os
from collections import defaultdict
from statistics import correlation as pearson

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = "data_openworld"
GRAPH_DIR = "graphs"
BIPARTITE_PATH = os.path.join(GRAPH_DIR, "bipartite_graph.graphml")
FRIENDS_PATH = os.path.join(DATA_DIR, "friend_edges.csv")
OUTPUT_PATH = os.path.join(GRAPH_DIR, "player_features.csv")
STATS_PATH = os.path.join(GRAPH_DIR, "feature_stats.json")

# Rounding for float features
ROUND = 6


def load_bipartite():
    G = nx.read_graphml(BIPARTITE_PATH)
    friends = pd.read_csv(FRIENDS_PATH, dtype=str)
    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, friends


def compute_features(G, friends):
    players = sorted(node for node, d in G.nodes(data=True) if d.get("type") == "player")
    player_set = set(players)
    n = len(players)
    log.info("Computing features for %d players (%d pairs)...", n, n * (n - 1) // 2)

    # Precompute per-player data
    game_sets = {}
    playtime_vec = {}
    player_degree = {}
    game_degree = {}
    for p in players:
        neigh = list(G.neighbors(p))
        game_sets[p] = set(neigh)
        playtime_vec[p] = {g: float(G[p][g].get("weight", 0.0)) for g in neigh}
        player_degree[p] = G.degree(p)
    for gnode, d in G.nodes(data=True):
        if d.get("type") == "game":
            game_degree[gnode] = G.degree(gnode)

    # Friend lookup
    friend_pairs = set()
    for _, row in friends.iterrows():
        p1, p2 = row["player1"], row["player2"]
        if p1 in player_set and p2 in player_set:
            friend_pairs.add(tuple(sorted((p1, p2))))

    rows = []
    for i in range(n):
        p1 = players[i]
        g1 = game_sets[p1]
        t1 = playtime_vec[p1]
        if i % 50 == 0:
            log.info("  Progress: %d / %d players", i, n)
        for j in range(i + 1, n):
            p2 = players[j]
            g2 = game_sets[p2]
            t2 = playtime_vec[p2]

            shared = g1 & g2
            union = g1 | g2
            n_shared = len(shared)

            # ---- Structural features ----

            # 1. Jaccard similarity
            jaccard = n_shared / len(union) if union else 0.0

            # 2. Weighted Jaccard (by playtime hours)
            sum_min = sum(min(t1.get(g, 0), t2.get(g, 0)) for g in shared)
            sum_max = sum(max(t1.get(g, 0), t2.get(g, 0)) for g in union)
            weighted_jaccard = sum_min / sum_max if sum_max > 0 else 0.0

            # 3. Common Neighbors (bipartite degree)
            common_neighbors = n_shared

            # 4. Cosine similarity (binary vectors)
            norm1 = math.sqrt(len(g1))
            norm2 = math.sqrt(len(g2))
            cosine = n_shared / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

            # ---- Graph-based features ----

            # 5. Adamic-Adar on the bipartite graph
            adamic_adar = sum(1.0 / math.log(max(2, float(game_degree.get(g, 2)))) for g in shared)

            # 6. Preferential Attachment (product of bipartite degrees)
            pa_score = player_degree[p1] * player_degree[p2]

            # ---- Behavioral features ----

            # 7. Playtime correlation across shared games
            playtime_corr = 0.0
            if n_shared >= 3:
                vals1 = [t1[g] for g in shared]
                vals2 = [t2[g] for g in shared]
                if max(vals1) > min(vals1) and max(vals2) > min(vals2):
                    playtime_corr = round(pearson(vals1, vals2), ROUND)

            # 8. Genre overlap (stub) — requires external genre mapping
            genre_overlap = -1.0  # -1 = not computed

            rows.append({
                "player1": p1,
                "player2": p2,
                "jaccard": round(jaccard, ROUND),
                "weighted_jaccard": round(weighted_jaccard, ROUND),
                "common_neighbors": n_shared,
                "cosine": round(cosine, ROUND),
                "adamic_adar": round(adamic_adar, ROUND),
                "preferential_attachment": pa_score,
                "playtime_correlation": playtime_corr,
                "genre_overlap": genre_overlap,
                "is_friend": (p1, p2) in friend_pairs,
            })

    df = pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)
    log.info("Done: %d player pairs, %d features", len(df), len(df.columns) - 2)
    return df


def compute_stats(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    for col in num_cols:
        vals = df[col].dropna()
        stats[col] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "p25": round(float(vals.quantile(0.25)), 4),
            "p50": round(float(vals.median()), 4),
            "p75": round(float(vals.quantile(0.75)), 4),
            "max": round(float(vals.max()), 4),
        }
    # Is_friend prevalence
    stats["is_friend"] = {
        "count_true": int(df["is_friend"].sum()),
        "count_false": int((~df["is_friend"]).sum()),
        "prevalence": round(float(df["is_friend"].mean()), 6),
    }
    return stats


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    log.info("Loading bipartite graph...")
    G, friends = load_bipartite()

    log.info("Computing all features...")
    df = compute_features(G, friends)

    log.info("Saving features...")
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved: %s (%d rows, %d cols)", OUTPUT_PATH, len(df), len(df.columns))

    log.info("Computing summary statistics...")
    stats = compute_stats(df)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved: %s", STATS_PATH)

    # Print summary
    log.info("")
    log.info("=== Feature Summary ===")
    for col in ["jaccard", "weighted_jaccard", "common_neighbors", "cosine", "adamic_adar", "preferential_attachment", "playtime_correlation"]:
        s = stats[col]
        log.info("  %-24s mean=%-10.4f  p50=%-10.4f  range=[%.4f, %.4f]", col, s["mean"], s["p50"], s["min"], s["max"])
    log.info("  %-24s friends=%d / total=%d (%.4f)", "is_friend", stats["is_friend"]["count_true"],
             stats["is_friend"]["count_true"] + stats["is_friend"]["count_false"], stats["is_friend"]["prevalence"])


if __name__ == "__main__":
    main()
