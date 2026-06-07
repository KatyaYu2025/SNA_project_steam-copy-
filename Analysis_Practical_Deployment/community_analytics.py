"""
Cross-Genre Influencer Identification and Community Analytics.

Identifies bridge players who connect disparate gaming communities,
computes influence scores, maps genre community structures,
and produces actionable insights for community managers.

Output:
  Analysis_Practical_Deployment/cross_genre_influencers.json
"""

import json
import logging
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROUND = 6
BRIDGE_PATH = "Analysis_Homophily_and_Genre_Bridges/bridge_players.csv"
EGO_PATH = "Analysis_Homophily_and_Genre_Bridges/bridge_ego_analysis.json"
PLAYERS_PATH = "data_openworld/players.csv"
PLAYER_GAMES_PATH = "data_openworld/player_games.csv"
GAME_GENRES_PATH = "data_openworld/game_genres.csv"
FEATURES_PATH = "graphs/player_features.csv"
FRIENDS_PATH = "data_openworld/friend_edges.csv"
OUTPUT_PATH = "Analysis_Practical_Deployment/cross_genre_influencers.json"
TOP_INFLUENCERS = 20
MIN_GENRE_PLAYERS = 10


def load_all_data() -> dict:
    """Load all input data files. Returns dict of DataFrames (CSV) and dicts (JSON)."""
    csv_files = {
        "bridge_players": BRIDGE_PATH,
        "players": PLAYERS_PATH,
        "player_games": PLAYER_GAMES_PATH,
        "game_genres": GAME_GENRES_PATH,
        "player_features": FEATURES_PATH,
        "friend_edges": FRIENDS_PATH,
    }
    json_files = {
        "ego_analysis": EGO_PATH,
    }

    data = {}
    for key, path in csv_files.items():
        if not os.path.exists(path):
            log.warning("File not found: %s", path)
            data[key] = pd.DataFrame()
            continue
        data[key] = pd.read_csv(path, dtype=str)
        log.info("Loaded %s: %d rows", path, len(data[key]))

    for key, path in json_files.items():
        if not os.path.exists(path):
            log.warning("File not found: %s", path)
            data[key] = {}
            continue
        with open(path) as f:
            data[key] = json.load(f)
        log.info("Loaded %s: %d entries", path, len(data[key]))

    return data


def compute_influence_scores(data: dict) -> pd.DataFrame:
    """Identify top bridge players and compute influence scores."""
    df_bridge = data.get("bridge_players", pd.DataFrame())
    if df_bridge.empty or "is_bridge" not in df_bridge.columns:
        log.warning("No bridge players data available")
        return pd.DataFrame()

    df_players = data.get("players", pd.DataFrame())
    ego = data.get("ego_analysis", {})
    df_friends = data.get("friend_edges", pd.DataFrame())

    df_bridge = df_bridge[df_bridge["is_bridge"] == "True"].copy()
    if df_bridge.empty:
        log.warning("No bridge players flagged as bridge")
        return pd.DataFrame()

    df_bridge["genre_entropy"] = pd.to_numeric(df_bridge["genre_entropy"])

    if not df_players.empty and "steamid" in df_players.columns:
        df_bridge = df_bridge.merge(
            df_players[["steamid", "friends_count"]], on="steamid", how="left"
        )
    df_bridge["friends_count"] = pd.to_numeric(df_bridge["friends_count"]).fillna(0)

    df_bridge["influence_score"] = (
        df_bridge["genre_entropy"] * np.log1p(df_bridge["friends_count"])
    )

    df_bridge["num_ego_clusters"] = 0
    df_bridge["silhouette_score"] = 0.0
    for i, row in df_bridge.iterrows():
        sid = row["steamid"]
        entry = ego.get(sid, {})
        clusters = entry.get("ego_clusters")
        if clusters:
            df_bridge.at[i, "num_ego_clusters"] = len(clusters)
        df_bridge.at[i, "silhouette_score"] = entry.get("silhouette_score", 0.0)

    df_influencers = df_bridge.sort_values(
        "genre_entropy", ascending=False
    ).head(TOP_INFLUENCERS).copy()

    bridge_ids = set(df_bridge["steamid"])
    friendships_with_bridges = {sid: 0 for sid in df_influencers["steamid"]}
    if not df_friends.empty and "player1" in df_friends.columns:
        for _, row in df_friends.iterrows():
            p1, p2 = row["player1"], row["player2"]
            if p1 in bridge_ids and p2 in bridge_ids:
                if p1 in friendships_with_bridges:
                    friendships_with_bridges[p1] += 1
                if p2 in friendships_with_bridges:
                    friendships_with_bridges[p2] += 1

    df_influencers["bridge_friends_count"] = (
        df_influencers["steamid"].map(friendships_with_bridges)
    )

    log.info(
        "Computed influence scores for %d influencers", len(df_influencers)
    )
    return df_influencers


def analyze_genre_communities(data: dict) -> dict:
    """Build genre community map: major genres, associated players, and bridge connections."""
    df_games = data.get("player_games", pd.DataFrame())
    df_genres = data.get("game_genres", pd.DataFrame())
    df_bridge = data.get("bridge_players", pd.DataFrame())

    if df_games.empty or df_genres.empty:
        return {}

    merged = df_games.merge(
        df_genres[["appid", "primary_genre"]], on="appid", how="left"
    )
    merged["primary_genre"] = merged["primary_genre"].fillna("Unknown")

    genre_player_counts = merged.groupby("primary_genre")["steamid"].nunique()
    major_genres = genre_player_counts[
        genre_player_counts >= MIN_GENRE_PLAYERS
    ].index.tolist()

    player_top_genre = {}
    for steamid, group in merged.groupby("steamid"):
        counts = group["primary_genre"].value_counts()
        if not counts.empty:
            player_top_genre[steamid] = counts.index[0]

    bridge_ids = set()
    if not df_bridge.empty and "is_bridge" in df_bridge.columns:
        bridge_ids = set(
            df_bridge[df_bridge["is_bridge"] == "True"]["steamid"]
        )

    genre_map = {}
    for genre in major_genres:
        primary_players = {
            pid for pid, pg in player_top_genre.items() if pg == genre
        }
        bridges_in_genre = primary_players & bridge_ids

        connections = defaultdict(list)
        for bid in bridges_in_genre:
            bridge_games = merged[merged["steamid"] == bid]
            other_genres = set(bridge_games["primary_genre"].unique()) - {genre}
            for og in other_genres:
                connections[og].append(bid)

        genre_map[genre] = {
            "player_count": len(primary_players),
            "primary_players_sample": sorted(primary_players)[:10],
            "bridge_players": sorted(bridges_in_genre),
            "connected_to": {
                g: sorted(set(v)) for g, v in sorted(connections.items())
            },
        }

    log.info(
        "Genre community map: %d major genres (>=%d players)",
        len(genre_map),
        MIN_GENRE_PLAYERS,
    )
    return genre_map


def generate_insights(
    influencers: pd.DataFrame, genre_communities: dict
) -> list[str]:
    """Generate human-readable insights for each top influencer."""
    insights = []
    for _, row in influencers.iterrows():
        sid = row["steamid"]
        entropy = row["genre_entropy"]
        friends_count = int(row["friends_count"])
        n_clusters = int(row["num_ego_clusters"])
        bridge_friends = int(row.get("bridge_friends_count", 0))
        influence_score = row["influence_score"]

        connected_genres = set()
        for genre, info in genre_communities.items():
            if sid in info.get("bridge_players", []):
                connected_genres.add(genre)
            for other_genre, bps in info.get("connected_to", {}).items():
                if sid in bps:
                    connected_genres.add(other_genre)

        genre_str = ", ".join(sorted(connected_genres)) if connected_genres else "various"

        cluster_note = (
            f"Their friends split into {n_clusters} distinct genre clusters."
            if n_clusters >= 2
            else "Their friends share similar genre preferences."
        )

        bridge_note = ""
        if bridge_friends > 0:
            bridge_note = (
                f" They are connected to {bridge_friends} other bridge player(s)."
            )

        insights.append(
            f"Player {sid} (entropy={entropy:.2f}) connects {genre_str} communities. "
            f"Has {friends_count} friends. {cluster_note}"
            f"{bridge_note} "
            f"Influence score: {influence_score:.2f}. "
            f"Would be a valuable community ambassador for cross-promotion."
        )

    return insights


def save_results(
    influencers: pd.DataFrame,
    genre_communities: dict,
    insights: list[str],
):
    """Save structured results to JSON and print summary table to stdout."""
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

    results = {
        "top_influencers": [],
        "genre_communities": genre_communities,
        "insights": insights,
    }

    for _, row in influencers.iterrows():
        results["top_influencers"].append({
            "steamid": row["steamid"],
            "genre_entropy": round(row["genre_entropy"], ROUND),
            "friends_count": int(row["friends_count"]),
            "influence_score": round(row["influence_score"], ROUND),
            "num_ego_clusters": int(row["num_ego_clusters"]),
            "silhouette_score": round(row["silhouette_score"], ROUND),
            "bridge_friends_count": int(row.get("bridge_friends_count", 0)),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", OUTPUT_PATH)

    print()
    print("=" * 100)
    print("  CROSS-GENRE INFLUENCER SUMMARY")
    print("=" * 100)
    header = f"{'SteamID':>18} {'Entropy':>8} {'Friends':>8} {'Influence':>10} {'Clusters':>9} {'BrFriends':>9}"
    print(header)
    print("-" * 100)
    for _, row in influencers.iterrows():
        print(
            f"{row['steamid']:>18} "
            f"{row['genre_entropy']:>8.4f} "
            f"{int(row['friends_count']):>8} "
            f"{row['influence_score']:>10.2f} "
            f"{int(row['num_ego_clusters']):>9} "
            f"{int(row.get('bridge_friends_count', 0)):>9}"
        )
    print("-" * 100)
    multi_cluster = sum(
        1 for _, r in influencers.iterrows() if int(r["num_ego_clusters"]) >= 2
    )
    print(f"\n  Top {len(influencers)} influencers identified.")
    print(f"  {multi_cluster} have friends in 2+ genre clusters.")
    print(f"  Insights saved to {OUTPUT_PATH}")
    print()


def main():
    log.info("Loading all data...")
    data = load_all_data()

    log.info("Computing influence scores...")
    influencers = compute_influence_scores(data)

    log.info("Analyzing genre communities...")
    genre_communities = analyze_genre_communities(data)

    if not influencers.empty:
        log.info("Generating insights...")
        insights = generate_insights(influencers, genre_communities)
        save_results(influencers, genre_communities, insights)
    else:
        log.warning("No influencers found; skipping insight generation.")


if __name__ == "__main__":
    main()
