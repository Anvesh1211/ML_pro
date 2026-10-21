import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/spending_behavior.csv")

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["anomaly"] = model.fit_predict(df[["amount"]])

df["anomaly_status"] = df["anomaly"].map({
    1: "Normal",
    -1: "Unusual"
})

unusual = df[df["anomaly"] == -1].sort_values(
    "amount", ascending=False
)

print("===== ANOMALY DETECTION =====")
print(f"Total transactions: {len(df)}")
print(f"Unusual transactions: {len(unusual)}")

print("\n===== UNUSUAL TRANSACTIONS =====")
print(
    unusual[
        ["transaction_date", "transaction_details", "amount",
         "predicted_category", "anomaly_status"]
    ].to_string(index=False)
)

df.to_csv("data/anomaly_results.csv", index=False)

print("\nAnomaly analysis saved.")