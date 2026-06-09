import pickle

def load_player_games():
    with open("player_games.pkl", "rb") as f:
        return pickle.load(f)

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def recommend(steamid, player_games, top_k=10):
    if steamid not in player_games:
        return []
    target_set = player_games[steamid]
    scores = []
    for pid, games in player_games.items():
        if pid == steamid:
            continue
        sim = jaccard(target_set, games)
        common = len(target_set & games)
        scores.append((pid, sim, common))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]