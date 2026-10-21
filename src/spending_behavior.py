import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/final_phonepe_results.csv")

expenses = df[df["transaction_type"] == "DEBIT"].copy()

features = expenses[["amount"]].copy()

scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
expenses["spending_cluster"] = kmeans.fit_predict(X)

cluster_summary = expenses.groupby("spending_cluster")["amount"].agg(
    ["count", "mean", "min", "max"]
).sort_values("mean")

print("===== SPENDING BEHAVIOUR CLUSTERS =====")
print(cluster_summary)

expenses.to_csv("data/spending_behavior.csv", index=False)

print("\nSpending behaviour analysis saved.")