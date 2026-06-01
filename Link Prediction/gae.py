"""
Graph Autoencoder (GAE) for Link Prediction.

Trains a GAE on the projected player graph to reconstruct friendship edges.
Uses PyTorch Geometric.

Encoder: 2-layer GCN → latent embeddings
Decoder: Inner product → reconstructed adjacency

Outputs:
  Link Prediction/gae_results.json          — AUC, Precision@k, Recall@k
  Link Prediction/gae_training_curve.png    — loss curve
"""

import json
import logging
import os
import sys
import warnings
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

DATA_PATH = "graphs/player_features.csv"
EDGE_LIST_PATH = "graphs/projected_edges_scored.csv"
OUT_DIR = Path("Link Prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "gae_results.json"
CURVE_PATH = OUT_DIR / "gae_training_curve.png"

EMBED_DIM = 64
HIDDEN_DIM = 128
LR = 0.01
EPOCHS = 200
RANDOM_STATE = 42
TOP_K_EDGES_PER_NODE = 20  # sparsify projected graph
TRAIN_RATIO = 0.8

K_VALUES = [10, 20, 50, 100, 200]


def build_projected_graph(df):
    """Build a sparse player-player graph from Jaccard-weighted edges."""
    players = sorted(set(df["player1"].unique()) | set(df["player2"].unique()))
    player_to_idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    log.info("Building projected graph: %d players", n)

    # For each player, keep top K edges by Jaccard similarity
    G = nx.Graph()
    G.add_nodes_from(players)

    edges_by_player = {p: [] for p in players}
    for _, row in df.iterrows():
        p1, p2 = row["player1"], row["player2"]
        w = row["jaccard"]
        edges_by_player[p1].append((p2, w))
        edges_by_player[p2].append((p1, w))

    edge_count = 0
    for p, edges in edges_by_player.items():
        edges.sort(key=lambda x: -x[1])
        for target, w in edges[:TOP_K_EDGES_PER_NODE]:
            if not G.has_edge(p, target):
                G.add_edge(p, target, weight=w)
                edge_count += 1

    log.info("  Sparse graph: %d nodes, %d edges", n, edge_count)
    return G, player_to_idx


def prepare_data(G, player_to_idx, df):
    """Create PyG Data object with train/test split."""
    import torch
    from sklearn.model_selection import train_test_split

    friends = df[df["is_friend"]].copy()
    non_friends = df[~df["is_friend"]].copy()

    # For GAE we use the edges from the projected graph + friendships
    # Positive edges = friendships
    pos_edges = list(friends[["player1", "player2"]].itertuples(index=False, name=None))
    pos_edges = [(player_to_idx[a], player_to_idx[b]) for a, b in pos_edges
                 if a in player_to_idx and b in player_to_idx]
    pos_edges = list(set(tuple(sorted(e)) for e in pos_edges))
    log.info("  Positive edges (friendships): %d", len(pos_edges))

    # Sample negative edges = non-friend pairs (balanced = same count as positive)
    rng = np.random.RandomState(RANDOM_STATE)
    neg_candidates = list(non_friends[["player1", "player2"]].itertuples(index=False, name=None))
    neg_indices = rng.choice(len(neg_candidates), min(len(neg_candidates), len(pos_edges) * 2), replace=False)
    neg_edges = []
    for idx in neg_indices:
        a, b = neg_candidates[idx]
        if a in player_to_idx and b in player_to_idx:
            neg_edges.append((player_to_idx[a], player_to_idx[b]))
    neg_edges = list(set(neg_edges))
    log.info("  Negative edges (sampled): %d", len(neg_edges))

    all_edges = pos_edges + neg_edges
    all_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

    train_edges, test_edges, train_labels, test_labels = train_test_split(
        all_edges, all_labels, test_size=1 - TRAIN_RATIO,
        stratify=all_labels, random_state=RANDOM_STATE
    )

    # Adjacency matrix from projected graph edges only (for GAE input)
    n = len(player_to_idx)
    edge_index_list = []
    for u, v in G.edges():
        i, j = player_to_idx[u], player_to_idx[v]
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Node features: try to load Node2Vec embeddings, fall back to identity
    n2v_path = OUT_DIR / "node2vec_embeddings.npy"
    if n2v_path.exists():
        emb = np.load(n2v_path).astype(np.float64)
        node_features = torch.tensor(emb, dtype=torch.float)
        log.info("  Using Node2Vec embeddings as node features (dim=%d)", emb.shape[1])
    else:
        node_features = torch.eye(n, dtype=torch.float)

    data = {
        "n": n,
        "edge_index": edge_index,
        "node_features": node_features,
        "train_edges": torch.tensor(train_edges, dtype=torch.long),
        "train_labels": torch.tensor(train_labels, dtype=torch.float),
        "test_edges": torch.tensor(test_edges, dtype=torch.long),
        "test_labels": torch.tensor(test_labels, dtype=torch.float),
    }
    log.info("  Train edges: %d, Test edges: %d", len(train_edges), len(test_edges))
    return data


def train_gae(data):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("  Device: %s", device)

    class GCNEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    class GAEModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x, edge_index):
            z = self.encoder(x, edge_index)
            return z

        def decode(self, z, edges):
            return (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=1).sigmoid()

        def recon_loss(self, z, pos_edge_index, neg_edge_index):
            pos_score = self.decode(z, pos_edge_index)
            neg_score = self.decode(z, neg_edge_index)
            loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
            loss += F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
            return loss

    in_dim = data["node_features"].shape[1]
    encoder = GCNEncoder(in_dim, HIDDEN_DIM, EMBED_DIM)
    model = GAEModel(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    edge_index = data["edge_index"].to(device)
    x = data["node_features"].to(device)
    train_edges = data["train_edges"].to(device)
    train_labels = data["train_labels"].to(device)

    # Separate positive and negative training edges
    pos_mask = train_labels == 1.0
    pos_train_edges = train_edges[pos_mask]
    neg_train_edges = train_edges[~pos_mask]
    n_pos_train = pos_train_edges.size(0)
    n_neg_train = neg_train_edges.size(0)
    log.info("    Train: %d pos edges, %d neg edges", n_pos_train, n_neg_train)

    all_nodes = torch.arange(data["n"], device=device)
    losses = []
    for epoch in range(EPOCHS):
        model.train()

        z = model(x, edge_index)

        # Decode positive edges
        if n_pos_train > 0:
            pos_score = model.decode(z, pos_train_edges)
            pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
        else:
            pos_loss = 0.0

        # Sample additional negative edges for balanced training
        neg_i = all_nodes[torch.randint(0, data["n"], (n_pos_train * 2,))]
        neg_j = all_nodes[torch.randint(0, data["n"], (n_pos_train * 2,))]
        neg_edges = torch.stack([neg_i, neg_j], dim=1)
        if n_neg_train > 0:
            neg_edges = torch.cat([neg_edges, neg_train_edges], dim=0)
        neg_score = model.decode(z, neg_edges)
        neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if epoch == 0:
            log.info("    DEBUG loss=%.10f pos_score mean=%.6f neg_score mean=%.6f",
                     loss_val, pos_score.mean().item() if isinstance(pos_loss, torch.Tensor) else 0,
                     neg_score.mean().item())
        if epoch % 20 == 0:
            log.info("    Epoch %3d | Loss: %.6f", epoch, loss_val)

    log.info("  Training complete. Final loss: %.6f", losses[-1])

    # Evaluate
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        test_edges = data["test_edges"].to(device)
        test_labels = data["test_labels"].to(device)

        # Get scores for both positive and negative test edges
        pos_test_mask = test_labels == 1.0
        neg_test_mask = ~pos_test_mask
        all_test_scores = []
        all_test_labels = []
        if pos_test_mask.sum() > 0:
            pos_test = test_edges[pos_test_mask]
            pos_scores = model.decode(z, pos_test).cpu().numpy()
            all_test_scores.extend(pos_scores.tolist())
            all_test_labels.extend([1] * len(pos_scores))
        if neg_test_mask.sum() > 0:
            neg_test = test_edges[neg_test_mask]
            neg_scores = model.decode(z, neg_test).cpu().numpy()
            all_test_scores.extend(neg_scores.tolist())
            all_test_labels.extend([0] * len(neg_scores))

    y_true = np.array(all_test_labels)
    y_score = np.array(all_test_scores)

    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc

    auc_val = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = pr_auc(rec, prec)

    results = {
        "model_params": {
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "lr": LR,
            "top_k_edges": TOP_K_EDGES_PER_NODE,
        },
        "auc_roc": round(auc_val, 4),
        "auc_pr": round(pr_auc_val, 4),
        "final_loss": round(float(losses[-1]), 6),
    }

    total_pos_test = int(y_true.sum())
    for k in K_VALUES:
        idx = np.argsort(y_score)[::-1][:k]
        results[f"precision@{k}"] = round(float(y_true[idx].mean()), 4)
        results[f"recall@{k}"] = round(float(y_true[idx].sum() / max(total_pos_test, 1)), 4)

    _plot_loss(losses)
    return results


def _plot_loss(losses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(losses, color="#3498db", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("GAE Training Loss")
        plt.tight_layout()
        plt.savefig(CURVE_PATH, dpi=150)
        log.info("Saved training curve to %s", CURVE_PATH)
    except Exception as e:
        log.warning("Could not generate loss plot: %s", e)


def main():
    try:
        import torch
        from torch_geometric.nn import GCNConv
    except ImportError:
        log.error("PyTorch Geometric not installed. Run: pip install torch_geometric")
        sys.exit(1)

    log.info("Loading features from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, dtype=str)
    df["is_friend"] = df["is_friend"] == "True"
    df["jaccard"] = pd.to_numeric(df["jaccard"])
    log.info("Loaded %d rows", len(df))

    G, player_to_idx = build_projected_graph(df)
    data = prepare_data(G, player_to_idx, df)
    results = train_gae(data)

    log.info("")
    log.info("=== GAE Results ===")
    log.info("  AUC-ROC = %.4f", results["auc_roc"])
    log.info("  AUC-PR  = %.4f", results["auc_pr"])
    log.info("  P@10    = %.4f", results["precision@10"])
    log.info("  P@50    = %.4f", results["precision@50"])
    log.info("  R@50    = %.4f", results["recall@50"])

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
