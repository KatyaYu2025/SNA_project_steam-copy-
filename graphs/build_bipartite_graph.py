"""
Build a weighted bipartite graph from Steam player-game data.

Nodes: players (bipartite=0) and games (bipartite=1)
Edges: player owns game, weighted by playtime_forever (hours)

Outputs (all → graphs/):
  - bipartite_graph.gexf        → Gephi (with viz:position for left-right layout)
  - bipartite_graph.graphml     → NetworkX / igraph
  - graph_stats.json            → summary + degree distribution
  - nodes.csv                   → all node attributes
  - projection_stats.csv        → player-player Jaccard similarity edges

Usage: python build_bipartite_graph.py
"""

import json
import logging
import os
from collections import Counter

import networkx as nx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = "data_openworld"
GRAPH_DIR = "graphs"
OUTPUT_STATS = os.path.join(GRAPH_DIR, "graph_stats.json")
OUTPUT_NODES = os.path.join(GRAPH_DIR, "nodes.csv")
OUTPUT_GEXF = os.path.join(GRAPH_DIR, "bipartite_graph.gexf")
OUTPUT_GRAPHML = os.path.join(GRAPH_DIR, "bipartite_graph.graphml")
OUTPUT_PROJECTION_STATS = os.path.join(GRAPH_DIR, "projection_stats.csv")

TARGET_APPS = {
    105600: "Terraria",
    1091500: "Cyberpunk 2077",
    1174180: "Red Dead Redemption 2",
    1245620: "ELDEN RING",
    252490: "Rust",
    264710: "Subnautica",
    271590: "Grand Theft Auto V Legacy",
    275850: "No Man's Sky",
    292030: "The Witcher 3: Wild Hunt",
    332200: "State of Decay 2",
    377160: "Fallout 4",
    413150: "Stardew Valley",
    489830: "The Elder Scrolls V: Skyrim Special Edition",
    534380: "Dying Light 2 Stay Human",
    990080: "Hogwarts Legacy",
}

# GEXF namespace with viz extension
NS_GEXF = "http://www.gexf.net/1.2draft"
NS_VIZ = "http://www.gexf.net/1.2draft/viz"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"

# Build from scratch as string to avoid ElementTree namespace headaches
# (viz:position requires namespace-aware serialization)

LAYOUT_X_PLAYER = -10000.0
LAYOUT_X_GAME = 10000.0
LAYOUT_HEIGHT = 30000.0


def load_data():
    games = pd.read_csv(os.path.join(DATA_DIR, "player_games.csv"))
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))
    friends = pd.read_csv(os.path.join(DATA_DIR, "friend_edges.csv"))
    log.info("Loaded %d player-game edges, %d players, %d friend edges", len(games), len(players), len(friends))
    return games, players, friends


def build_graph(games_df):
    G = nx.Graph()

    steamids = games_df["steamid"].unique()
    appids = set(games_df["appid"].unique())

    target_present = sorted(appids & TARGET_APPS.keys())
    log.info("Target games found in data: %d / %d", len(target_present), len(TARGET_APPS))

    G.add_nodes_from(steamids, bipartite=0, type="player")

    game_names = dict(zip(games_df["appid"], games_df["game_name"]))
    for appid in appids:
        G.add_node(
            appid,
            bipartite=1,
            type="game",
            game_name=game_names.get(appid, str(appid)),
            is_target=appid in TARGET_APPS,
        )

    edges = []
    weight_sum = 0.0
    for _, row in games_df.iterrows():
        w = max(row["playtime_forever"] / 60.0, 0.01)
        edges.append((row["steamid"], row["appid"], {"weight": round(w, 2)}))
        weight_sum += w

    G.add_edges_from(edges)

    log.info(
        "Graph: %d nodes, %d edges, avg weight %.1f h",
        G.number_of_nodes(),
        G.number_of_edges(),
        weight_sum / len(edges) if edges else 0,
    )
    return G, int(weight_sum)


def compute_layout(G):
    player_nodes = sorted(
        [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0],
        key=lambda n: G.degree(n),
        reverse=True,
    )
    game_nodes = sorted(
        [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1],
        key=lambda n: G.degree(n),
        reverse=True,
    )

    positions = {}

    n_players = len(player_nodes)
    for i, n in enumerate(player_nodes):
        y = (i / max(n_players - 1, 1) - 0.5) * LAYOUT_HEIGHT
        positions[n] = (LAYOUT_X_PLAYER, round(y, 2))

    n_games = len(game_nodes)
    for i, n in enumerate(game_nodes):
        y = (i / max(n_games - 1, 1) - 0.5) * LAYOUT_HEIGHT
        positions[n] = (LAYOUT_X_GAME, round(y, 2))

    return positions


def compute_player_jaccard_similarities(G, friends_df):
    player_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    player_set = set(player_nodes)

    friend_pairs = set()
    for _, row in friends_df.iterrows():
        p1, p2 = row["player1"], row["player2"]
        if p1 in player_set and p2 in player_set:
            friend_pairs.add(tuple(sorted((p1, p2))))

    player_games = {p: set(G.neighbors(p)) for p in player_nodes}

    rows = []
    p_list = list(player_nodes)
    for i in range(len(p_list)):
        p1 = p_list[i]
        games1 = player_games[p1]
        for j in range(i + 1, len(p_list)):
            p2 = p_list[j]
            games2 = player_games[p2]
            intersection = len(games1 & games2)
            union = len(games1 | games2)
            if union == 0:
                continue
            jaccard = intersection / union
            rows.append({
                "player1": p1,
                "player2": p2,
                "jaccard_similarity": round(jaccard, 6),
                "shared_games": intersection,
                "is_friend": (p1, p2) in friend_pairs,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("jaccard_similarity", ascending=False).reset_index(drop=True)
    log.info("Player-player Jaccard pairs: %d (%.2f%% of possible)", len(df),
              len(df) / (len(player_nodes) * (len(player_nodes) - 1) / 2) * 100)
    return df


def compute_stats(G, total_weight):
    player_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    game_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    player_degrees = [d for _, d in G.degree(player_nodes)]
    game_degrees = [d for _, d in G.degree(game_nodes)]

    target_game_nodes = [n for n in game_nodes if G.nodes[n].get("is_target")]

    game_counts = Counter()
    for p in player_nodes:
        for g in G.neighbors(p):
            game_counts[g] += 1

    top_games = [
        {"appid": int(g), "name": G.nodes[g].get("game_name", ""), "player_count": c}
        for g, c in game_counts.most_common(20)
    ]

    def degree_distribution(degrees):
        from collections import Counter as Ctr
        dist = Ctr(degrees)
        return {
            "mean": round(float(pd.Series(degrees).mean()), 2),
            "median": int(pd.Series(degrees).median()),
            "std": round(float(pd.Series(degrees).std()), 2),
            "max": max(degrees) if degrees else 0,
            "min": min(degrees) if degrees else 0,
            "histogram": {str(k): v for k, v in sorted(dist.items())},
        }

    stats = {
        "summary": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "total_weight_hours": total_weight,
            "player_nodes": len(player_nodes),
            "game_nodes": len(game_nodes),
            "target_game_nodes": len(target_game_nodes),
            "density": round(nx.density(G), 6),
        },
        "degree_distribution": {
            "players": degree_distribution(player_degrees),
            "games": degree_distribution(game_degrees),
        },
        "top_games_by_players": top_games,
    }
    return stats


def save_nodes_csv(G, path):
    rows = []
    for n, d in G.nodes(data=True):
        if d.get("bipartite") == 0:
            rows.append({
                "id": n,
                "type": "player",
                "bipartite_set": 0,
                "label": f"Player_{str(n)[-6:]}",
                "game_name": "",
                "is_target": False,
                "degree": G.degree(n),
            })
        else:
            rows.append({
                "id": n,
                "type": "game",
                "bipartite_set": 1,
                "label": d.get("game_name", str(n)),
                "game_name": d.get("game_name", ""),
                "is_target": d.get("is_target", False),
                "degree": G.degree(n),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info("Saved nodes CSV (%d rows): %s", len(df), path)


def write_gexf_with_layout(G, positions, path):
    """Write GEXF with viz:position (players left, games right)."""
    import gzip
    from lxml import etree

    NSMAP = {None: NS_GEXF, "viz": NS_VIZ}

    root = etree.Element(f"{{{NS_GEXF}}}gexf", nsmap=NSMAP, version="1.2")
    meta = etree.SubElement(root, f"{{{NS_GEXF}}}meta", lastmodifieddate="2026-05-25")
    etree.SubElement(meta, f"{{{NS_GEXF}}}creator").text = "build_bipartite_graph.py"

    graph = etree.SubElement(root, f"{{{NS_GEXF}}}graph", defaultedgetype="undirected", mode="static")

    attrs = etree.SubElement(graph, f"{{{NS_GEXF}}}attributes", **{"class": "node", "mode": "static"})
    for aid, title, atype in [("0","bipartite","long"),("1","type","string"),("2","game_name","string"),("3","is_target","boolean")]:
        a = etree.SubElement(attrs, f"{{{NS_GEXF}}}attribute", id=aid, title=title, type=atype)

    nodes_elem = etree.SubElement(graph, f"{{{NS_GEXF}}}nodes")
    for n, d in G.nodes(data=True):
        label = f"P_{str(n)[-6:]}" if d.get("type") == "player" else d.get("game_name", str(n))[:40]
        nd = etree.SubElement(nodes_elem, f"{{{NS_GEXF}}}node", id=str(n), label=label)
        x, y = positions.get(n, (0.0, 0.0))
        etree.SubElement(nd, f"{{{NS_VIZ}}}position", x=str(x), y=str(y), z="0.0")
        av = etree.SubElement(nd, f"{{{NS_GEXF}}}attvalues")
        for aid, key in [("0","bipartite"),("1","type"),("2","game_name"),("3","is_target")]:
            val = d.get(key, "")
            val = str(val).lower() if isinstance(val, bool) else str(val)
            etree.SubElement(av, f"{{{NS_GEXF}}}attvalue", for_=aid, value=val)

    edges_elem = etree.SubElement(graph, f"{{{NS_GEXF}}}edges")
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        etree.SubElement(edges_elem, f"{{{NS_GEXF}}}edge",
                         id=str(i), source=str(u), target=str(v),
                         weight=str(d.get("weight", 1.0)))

    etree.indent(root, space="  ")
    xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)

    with open(path, "wb") as f:
        f.write(xml_bytes)

    gz_path = path + ".gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(xml_bytes)

    mb = os.path.getsize(path) / 1e6
    gz_mb = os.path.getsize(gz_path) / 1e6
    log.info("Saved GEXF (%.1f MB, .gz: %.1f MB): %s", mb, gz_mb, path)


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    log.info("Loading data...")
    games_df, players_df, friends_df = load_data()

    log.info("Building bipartite graph...")
    G, total_weight = build_graph(games_df)

    log.info("Computing layout positions (players left, games right)...")
    positions = compute_layout(G)

    log.info("Computing stats...")
    stats = compute_stats(G, total_weight)

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved stats: %s", OUTPUT_STATS)

    save_nodes_csv(G, OUTPUT_NODES)

    write_gexf_with_layout(G, positions, OUTPUT_GEXF)

    nx.write_graphml(G, OUTPUT_GRAPHML)
    log.info("Saved GraphML: %s", OUTPUT_GRAPHML)

    log.info("Computing player-player Jaccard similarities...")
    proj_df = compute_player_jaccard_similarities(G, friends_df)
    proj_df.to_csv(OUTPUT_PROJECTION_STATS, index=False)
    log.info("Saved projection stats (%d pairs): %s", len(proj_df), OUTPUT_PROJECTION_STATS)

    log.info("Done.")


if __name__ == "__main__":
    main()
