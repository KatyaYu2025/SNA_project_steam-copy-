## **Predicting friend connections based on shared game libraries on Steam using a homophily graph**

**Прогнозирование дружеских связей на основе общих библиотек игр в Steam с применением графа гомофилии**

### **Description**

Online gaming communities are among the largest and most socially active digital ecosystems, yet the mechanisms that drive friendship formation remain incompletely understood from a network science perspective. Every Steam user owns a collection of games --- and taken together, thousands of player libraries form a rich bipartite graph where players and games are nodes, and \"owns\" relationships are edges. This graph encodes not only individual preferences but also implicit social affinity: players who share many games are more likely to be friends, a phenomenon rooted in homophily --- the tendency to bond with others who have similar interests.

This project proposes to analyze and exploit the Player-Game graph using graph-based link prediction methods. The core tasks include: constructing a player--player projection weighted by Jaccard similarity of game libraries, predicting friendship edges from game overlap alone, evaluating prediction accuracy against real Steam friend lists, and identifying \"bridge players\" who connect entirely different gaming genres. By combining structural graph features with behavioral signals (playtime), the project bridges social network analysis, behavioral data mining, and recommender systems.

### **Implementation Steps**

1.  **Data Collection and Graph Construction**

-   **Source**:\
    > Steam Web API.

-   **Sample selection**:\
    > Collect N players (approximately 300-500 players) by crawling public Steam groups to ensure genre diversity.

-   **Data per player**:

    -   List of friends (SteamIDs).

    -   List of owned games (appid, playtime_forever).

-   **Construct a bipartite graph**:

    -   Nodes:\
        > players and games (two disjoint sets).

    -   Edges:\
        > \"player owns game X\", optionally weighted by total playtime (to distinguish active players from collectors).

-   **Project onto player--player graph**:

    -   Edge weight between two players.

    -   Compare edges with real friendships from friends lists.

2.  **Feature Engineering**

**For each pair of players (i, j):**

-   **Structural features:**

    -   Jaccard similarity (game overlap).

    -   Weighted Jaccard using playtime.

    -   Common neighbors in bipartite graph.

    -   Cosine similarity of binary game vectors.

-   **Graph-based features:**

    -   Adamic-Adar index on projected graph.

    -   Preferential attachment score.

-   **Behavioral features:**

    -   Genre overlap.

    -   Playtime correlation across shared games.

3.  **Link Prediction:**

-   **Task:\
    > **Binary classification --- given a pair of players (with no direct friendship edge in training), predict whether they are friends.

-   **Train / test split:**

    -   Positive edges --- real friendships.

    -   Negative edges --- random pairs of non-friends.

-   **Models to compare:**

    -   Heuristic baselines: Jaccard threshold for friend predicting.

    -   Node2Vec embeddings of players (trained on bipartite graph) + cosine similarity.

    -   Graph Autoencoder (GAE) on projected player graph.

4.  **Analysis of Homophily and Genre Bridges**

-   **Homophily analysis:**

    -   Compute average game overlap among friends vs. non-friends (distribution plots).

    -   Stratify by player activity (playtime quartiles).

-   **Genre-level homophily:**

    -   Map each game to primary genre.

    -   Compute genre similarity between friends and non-friends.

-   **Bridge players:**

    -   Identify players genre-diverse players, visualise them and see if their friends can be split into two or more separate communities (via genre of games they play).

### **Analysis and Practical Deployment**

-   **Analysis:**\
    > identify genre-diverse \"bridge players\" via entropy scoring and visualize them as distinct nodes in the projected graph.

-   **Practical Deployment:**\
    > possible browser extension that suggests potential friends based on game library overlap, or integrate into gaming community analytics to detect cross-genre influencers.

### **Expected Results**

The project aims to determine whether friendship in gaming networks can be predicted from game library overlap, using social network analysis and link prediction. Anticipated outcomes include:

1.  **Quantified predictive power of homophily ---** establishing baseline accuracy (AUC-ROC) of Jaccard similarity for friendship prediction, and demonstrating improvement by incorporating playtime and genre features.

2.  **Identification of bridge players ---** discovering individuals who connect disparate gaming genres, visualized as critical ties in the player graph.

3.  **A practical and interpretable friend recommender for gaming platforms** to suggest potential friends based solely on shared game libraries, without requiring existing social graph data.

This framework provides enhanced capabilities for understanding interest-driven social tie formation, improving social recommendation systems, and analyzing cross-community information flow in online gaming ecosystems.
