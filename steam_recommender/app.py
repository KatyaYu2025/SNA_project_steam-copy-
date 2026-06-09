import streamlit as st
import pandas as pd
import os
import pickle

# --- 自动预处理数据 ---
DATA_CSV = "player_games.csv"
PKL_FILE = "player_games.pkl"

# 如果 .pkl 文件不存在，或者 .csv 文件比它更新，就重新生成
if not os.path.exists(PKL_FILE) or os.path.getmtime(DATA_CSV) > os.path.getmtime(PKL_FILE):
    with st.spinner("正在初始化数据（首次运行需要一点时间）..."):
        df = pd.read_csv(DATA_CSV)
        # 只保留目标游戏
        df_target = df[df["is_target"] == 1]
        player_games = df_target.groupby("steamid")["appid"].apply(set).to_dict()
        with open(PKL_FILE, "wb") as f:
            pickle.dump(player_games, f)
        print("数据预处理完成！")

# 现在导入推荐函数
from recommender import load_player_games, recommend

st.set_page_config(page_title="Steam Friend Recommender", layout="centered")
st.title("🎮 Steam Friend Recommender")
st.markdown("Select your Steam ID, and the system will recommend potential friends based on game library overlap.")

player_games = load_player_games()
player_list = sorted(player_games.keys())
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