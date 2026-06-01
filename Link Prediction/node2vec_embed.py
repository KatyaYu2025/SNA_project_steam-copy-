"""
Node2Vec Link Prediction.

Trains Node2Vec embeddings on the bipartite player-game graph,
then uses cosine similarity between player embedding vectors as
a feature for link prediction.

Outputs:
  Link Prediction/node2vec_results.json — AUC, Precision@k, Recall@k
  Link Prediction/node2vec_embeddings.npy — player embedding matrix
  Link Prediction/embedding_visualization.png — 2D UMAP projection of player embeddings
"""

import json
import logging
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GRAPH_PATH = "graphs/bipartite_graph.graphml"
DATA_PATH = "graphs/player_features.csv"
OUT_DIR = Path("Link Prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "node2vec_results.json"
EMBED_PATH = OUT_DIR / "node2vec_embeddings.npy"
VIZ_PATH = OUT_DIR / "embedding_visualization.png"
PLAYER_MAP_PATH = OUT_DIR / "player_id_to_index.json"

EMBED_DIM = 64
WALK_LENGTH = 15
NUM_WALKS = 15
P = 1.0
Q = 1.0
WORKERS = 4
RANDOM_STATE = 42

K_VALUES = [10, 20, 50, 100, 200]


def train_node2vec(G):
    log.info("Running Node2Vec on bipartite graph (%d nodes, %d edges)...",
             G.number_of_nodes(), G.number_of_edges())

    from node2vec import Node2Vec

    n2v = Node2Vec(
        G,
        dimensions=EMBED_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=P,
        q=Q,
        workers=WORKERS,
        seed=RANDOM_STATE,
    )
    log.info("  Walks generated, training Word2Vec...")
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    log.info("  Node2Vec training complete (dim=%d)", EMBED_DIM)
    return model


def evaluate_embeddings(model, players, df):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc

    player_to_index = {pid: idx for idx, pid in enumerate(players)}

    # Extract embedding matrix for all players
    embedding_matrix = np.zeros((len(players), EMBED_DIM), dtype=np.float64)
    for pid, idx in player_to_index.items():
        try:
            embedding_matrix[idx] = model.wv[pid].astype(np.float64)
        except KeyError:
            log.warning("Player %s not in Node2Vec model, using zeros", pid)
            embedding_matrix[idx] = 0.0

    np.save(EMBED_PATH, embedding_matrix)
    log.info("Saved embeddings to %s (shape=%s)", EMBED_PATH, embedding_matrix.shape)

    with open(PLAYER_MAP_PATH, "w") as f:
        json.dump(player_to_index, f)
    log.info("Saved player ID map to %s", PLAYER_MAP_PATH)

    # Compute cosine similarity for each pair
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embedding_matrix / norms
    similarity_matrix = normalized @ normalized.T

    y_true = df["is_friend"].astype(int).values
    total_pos = y_true.sum()
    y_score = np.zeros(len(df))

    for idx, row in df.iterrows():
        p1, p2 = row["player1"], row["player2"]
        if p1 in player_to_index and p2 in player_to_index:
            i, j = player_to_index[p1], player_to_index[p2]
            y_score[idx] = float(similarity_matrix[i, j])

    auc_val = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = pr_auc(rec, prec)

    results = {
        "model_params": {
            "embed_dim": EMBED_DIM,
            "walk_length": WALK_LENGTH,
            "num_walks": NUM_WALKS,
            "p": P,
            "q": Q,
        },
        "auc_roc": round(auc_val, 4),
        "auc_pr": round(pr_auc_val, 4),
    }
    for k in K_VALUES:
        idx = np.argsort(y_score)[::-1][:k]
        results[f"precision@{k}"] = round(float(y_true[idx].mean()), 4)
        results[f"recall@{k}"] = round(float(y_true[idx].sum() / max(total_pos, 1)), 4)

    return results


def visualize_embeddings(embedding_matrix, df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        player_to_index = json.load(open(PLAYER_MAP_PATH))

        # Find a few players with high/low friend counts for visualization
        players_list = list(player_to_index.keys())
        is_friend_counts = {}
        for p in players_list:
            cnt = df[(df["player1"] == p) | (df["player2"] == p)]["is_friend"].sum()
            is_friend_counts[p] = int(cnt)

        n_players = len(players_list)
        if n_players > 1000:
            rng = np.random.RandomState(RANDOM_STATE)
            subset = rng.choice(n_players, 500, replace=False)
            players_list = [players_list[i] for i in subset]
        else:
            subset = slice(None)

        emb_subset = embedding_matrix[subset]
        friend_vals = np.array([is_friend_counts[p] for p in players_list])

        # UMAP or PCA for 2D viz
        try:
            import umap
            reducer = umap.UMAP(random_state=RANDOM_STATE)
            emb_2d = reducer.fit_transform(emb_subset)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=RANDOM_STATE)
            emb_2d = reducer.fit_transform(emb_subset)
            method = "PCA"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sc = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=friend_vals,
                             cmap="RdYlBu", s=10, alpha=0.7, edgecolors="none")
        axes[0].set_title(f"Player Embeddings ({method}) — colored by friend count")
        plt.colorbar(sc, ax=axes[0], label="Number of friends in sample")

        # Color by whether they have at least one friend
        has_friend = (friend_vals > 0).astype(int)
        colors = ["#3498db" if v else "#95a5a6" for v in has_friend]
        axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=10, alpha=0.7, edgecolors="none")
        axes[1].set_title(f"Player Embeddings ({method}) — has friend (blue) / no friend (gray)")

        plt.tight_layout()
        plt.savefig(VIZ_PATH, dpi=150)
        log.info("Saved embedding visualization to %s", VIZ_PATH)
    except Exception as e:
        log.warning("Could not generate visualization: %s", e)


def main():
    log.info("Loading graph from %s", GRAPH_PATH)
    G = nx.read_graphml(GRAPH_PATH)

    log.info("Loading features from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, dtype=str)
    df["is_friend"] = df["is_friend"] == "True"
    log.info("Loaded %d player pairs", len(df))

    players = sorted(n for n, d in G.nodes(data=True) if d.get("type") == "player")
    log.info("Found %d players in graph", len(players))

    model = train_node2vec(G)
    results = evaluate_embeddings(model, players, df)

    log.info("")
    log.info("=== Node2Vec Results ===")
    log.info("  AUC-ROC = %.4f", results["auc_roc"])
    log.info("  AUC-PR  = %.4f", results["auc_pr"])
    log.info("  P@10    = %.4f", results["precision@10"])
    log.info("  P@50    = %.4f", results["precision@50"])
    log.info("  R@50    = %.4f", results["recall@50"])

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", RESULTS_PATH)

    embed_matrix = np.load(EMBED_PATH)
    visualize_embeddings(embed_matrix, df)


if __name__ == "__main__":
    main()
