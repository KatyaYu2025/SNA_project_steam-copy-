import sys
import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_PAIRS = "graphs/player_features.csv"
DATA_PLAYERS = "data_openworld/players.csv"
DATA_BRIDGES = "Analysis_Homophily_and_Genre_Bridges/bridge_players.csv"

WEIGHT_JACCARD = 0.40
WEIGHT_ADAMIC_ADAR = 0.25
WEIGHT_GENRE_OVERLAP = 0.20
WEIGHT_PLAYTIME_CORR = 0.15

NUMERIC_COLS_PAIRS = [
    "jaccard", "weighted_jaccard", "common_neighbors", "cosine",
    "adamic_adar", "preferential_attachment", "playtime_correlation",
    "genre_overlap",
]
NUMERIC_COLS_PLAYERS = ["target_games_count", "avg_target_playtime_h", "friends_count", "total_games"]


def load_data():
    df_pairs = pd.read_csv(DATA_PAIRS, dtype=str)
    for col in NUMERIC_COLS_PAIRS:
        df_pairs[col] = pd.to_numeric(df_pairs[col], errors="coerce")
        if col == "genre_overlap":
            df_pairs[col] = df_pairs[col].clip(lower=-1.0, upper=1.0)

    df_players = pd.read_csv(DATA_PLAYERS, dtype=str)
    for col in NUMERIC_COLS_PLAYERS:
        df_players[col] = pd.to_numeric(df_players[col], errors="coerce")

    try:
        df_bridges = pd.read_csv(DATA_BRIDGES, dtype=str)
    except FileNotFoundError:
        logging.warning("bridge_players.csv not found — bridge info will be unavailable")
        df_bridges = pd.DataFrame(columns=["steamid", "genre_entropy", "is_bridge"])
    return df_pairs, df_players, df_bridges


def get_recommendations(df_pairs, df_players, df_bridges, target_id, top_n=10):
    mask = (df_pairs["player1"] == target_id) | (df_pairs["player2"] == target_id)
    target_pairs = df_pairs[mask].copy()
    if target_pairs.empty:
        logging.warning("No pairs found for player %s", target_id)
        return []

    other_id = np.where(
        target_pairs["player1"] == target_id,
        target_pairs["player2"],
        target_pairs["player1"],
    )
    target_pairs["other_id"] = other_id

    already_friends = set(target_pairs.loc[target_pairs["is_friend"] == "True", "other_id"])

    candidates = target_pairs[target_pairs["is_friend"] != "True"].copy()

    max_adamic_adar = candidates["adamic_adar"].max()
    if max_adamic_adar > 0:
        candidates["adamic_adar_norm"] = candidates["adamic_adar"] / max_adamic_adar
    else:
        candidates["adamic_adar_norm"] = 0.0

    candidates["playtime_corr_clamped"] = candidates["playtime_correlation"].clip(lower=0, upper=1)

    candidates["score"] = (
        WEIGHT_JACCARD * candidates["jaccard"]
        + WEIGHT_ADAMIC_ADAR * candidates["adamic_adar_norm"]
        + WEIGHT_GENRE_OVERLAP * candidates["genre_overlap"]
        + WEIGHT_PLAYTIME_CORR * candidates["playtime_corr_clamped"]
    )

    candidates = candidates.sort_values("score", ascending=False).head(top_n)

    bridge_map = {}
    if not df_bridges.empty and "steamid" in df_bridges.columns:
        for _, row in df_bridges.iterrows():
            bridge_map[row["steamid"]] = row.get("is_bridge", "False") == "True"

    recommendations = []
    for _, row in candidates.iterrows():
        rec = {
            "player_id": row["other_id"],
            "score": round(row["score"], 4),
            "jaccard": round(row["jaccard"], 4),
            "genre_overlap": round(row["genre_overlap"], 4),
            "is_bridge": bridge_map.get(row["other_id"], None),
        }
        recommendations.append(rec)

    player_row = df_players[df_players["steamid"] == target_id]
    player_info = {}
    if not player_row.empty:
        player_info = {
            "target_games_count": int(player_row.iloc[0]["target_games_count"]),
            "friends_count": int(player_row.iloc[0]["friends_count"]),
        }
    else:
        logging.warning("Player %s not found in players.csv", target_id)

    result = {
        "recommendations": recommendations,
        "player_info": player_info,
        "already_friends_count": len(already_friends),
        "already_friends": sorted(already_friends),
    }
    return result


def print_recommendations(result, target_id):
    recs = result["recommendations"]
    player_info = result["player_info"]
    already_friends_count = result["already_friends_count"]
    already_friends = result["already_friends"]

    print(f"\n{'=' * 80}")
    print(f"  Friend Recommendations for Steam ID: {target_id}")
    print(f"{'=' * 80}")

    if player_info:
        print(f"  Target Games: {player_info['target_games_count']}  |  "
              f"Friends: {player_info['friends_count']}  |  "
              f"Already have {already_friends_count} friend(s) — excluded from results")
    else:
        print(f"  (Player metadata not found)  |  Already have {already_friends_count} friend(s)")

    if already_friends:
        print(f"  Existing friends: {', '.join(list(already_friends)[:5])}{'...' if len(already_friends) > 5 else ''}")

    print(f"\n{'─' * 80}")
    print(f"  {'Rank':<6} {'Player ID':<22} {'Score':<10} {'Jaccard':<10} {'Genre Overlap':<14} {'Bridge?'}")
    print(f"{'─' * 80}")

    if not recs:
        print("  No recommendations available.")
    else:
        for i, rec in enumerate(recs, 1):
            bridge_str = (
                "Yes" if rec["is_bridge"] is True
                else "No" if rec["is_bridge"] is False
                else "N/A"
            )
            print(
                f"  {i:<6} {rec['player_id']:<22} {rec['score']:<10.4f} "
                f"{rec['jaccard']:<10.4f} {rec['genre_overlap']:<14.4f} {bridge_str}"
            )

    print(f"{'─' * 80}\n")


def main():
    df_pairs, df_players, df_bridges = load_data()

    all_players = sorted(set(df_pairs["player1"]).union(set(df_pairs["player2"])))

    if len(sys.argv) > 1:
        target_id = sys.argv[1]
    else:
        print("Steam Player ID not provided.")
        print("Example player IDs from the dataset:")
        for pid in all_players[:10]:
            print(f"  {pid}")
        target_id = input("Enter a Steam64 player ID: ").strip()
        if not target_id:
            logging.error("No player ID entered. Exiting.")
            sys.exit(1)

    if target_id not in all_players:
        logging.error("Player ID %s not found in the dataset.", target_id)
        sys.exit(1)

    result = get_recommendations(df_pairs, df_players, df_bridges, target_id)
    print_recommendations(result, target_id)


if __name__ == "__main__":
    main()
