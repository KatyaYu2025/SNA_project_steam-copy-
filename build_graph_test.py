import pandas as pd
import networkx as nx

# 读数据
file_path = r"D:\高经学习\研一下学期（M3-M4）\社会网络分析\player_games.csv"
df = pd.read_csv(file_path)

# 建图
B = nx.Graph()
player_nodes = df['steamid'].unique().tolist()
game_nodes = df['appid'].unique().tolist()
B.add_nodes_from(player_nodes, bipartite=0)
B.add_nodes_from(game_nodes, bipartite=1)

# 加边
edges = list(zip(df['steamid'], df['appid'], df['playtime_forever']))
B.add_weighted_edges_from(edges)

print(f"玩家数量: {len(player_nodes)}")
print(f"游戏数量: {len(game_nodes)}")
print(f"总节点数: {B.number_of_nodes()}")
print(f"总边数: {B.number_of_edges()}")

# 保存图
nx.write_edgelist(B, "player_game_bipartite.adjlist", data=['weight'])