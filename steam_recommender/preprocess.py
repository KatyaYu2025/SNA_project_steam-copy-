import pandas as pd
import pickle

# 读取数据
df = pd.read_csv("player_games.csv")

# 只保留目标游戏（is_target == 1）
df_target = df[df["is_target"] == 1]

# 构建玩家 -> 游戏集合
player_games = df_target.groupby("steamid")["appid"].apply(set).to_dict()

# 保存为 pickle 文件
with open("player_games.pkl", "wb") as f:
    pickle.dump(player_games, f)

print(f"预处理完成！共 {len(player_games)} 名玩家")