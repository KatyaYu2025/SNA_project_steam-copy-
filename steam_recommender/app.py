import streamlit as st
import pandas as pd
from recommender import load_player_games, recommend

st.set_page_config(page_title="Steam Friend Recommender", layout="centered")
st.title("🎮 Steam Friend Recommender")
st.markdown("Select your Steam ID, and the system will recommend potential friends based on game library overlap.")

# Load data
player_games = load_player_games()
player_list = sorted(player_games.keys())

# Display short ID (last 6 digits)
display_dict = {pid: f"...{str(pid)[-6:]}" for pid in player_list}

selected_display = st.selectbox("Your Steam ID", list(display_dict.values()))
selected_steamid = [pid for pid, disp in display_dict.items() if disp == selected_display][0]

if st.button("🔍 Recommend Friends"):
    with st.spinner("Computing..."):
        recs = recommend(selected_steamid, player_games, top_k=10)
    if recs:
        df = pd.DataFrame(recs, columns=["SteamID", "Similarity (Jaccard)", "Common Games"])
        df["Display ID"] = df["SteamID"].apply(lambda x: str(x)[-6:])
        df = df[["Display ID", "Similarity (Jaccard)", "Common Games"]]
        df["Similarity (Jaccard)"] = df["Similarity (Jaccard)"].round(4)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No recommendations found.")