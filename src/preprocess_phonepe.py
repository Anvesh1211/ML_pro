import pandas as pd
import re

df = pd.read_csv("data/testing.csv", skiprows=2)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df["transaction_date"] = df["Date"].dt.strftime("%Y-%m-%d")
df["transaction_time"] = df["Time"]
df["transaction_id"] = df["Transaction ID"]
df["utr"] = df["UTR"]
df["transaction_type"] = df["Transaction Type"]
df["amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df["currency"] = "INR"
df["transaction_details"] = df["Transaction Details"]

df["merchant"] = df["Transaction Details"].str.replace(
    r"^(Paid to|Received from)\s+", "", regex=True
).str.strip()

df["cashflow_type"] = df["transaction_type"]
df["is_expense"] = (df["transaction_type"] == "DEBIT").astype(int)

df["day_of_week"] = pd.to_datetime(df["Date"]).dt.day_name()
df["month"] = pd.to_datetime(df["Date"]).dt.month
df["year"] = pd.to_datetime(df["Date"]).dt.year

df["payment_method"] = "UPI"

df["transaction_status"] = "SUCCESS"
df["city"] = "Pune"
df["notes"] = ""

def classify(text):
    text = str(text).lower()

    if any(x in text for x in ["swiggy", "zomato", "hotel", "restaurant", "dining", "pani puri", "food"]):
        return "Food & Dining"
    if any(x in text for x in ["super market", "traders", "fruit", "vegetable", "agro"]):
        return "Grocery"
    if "recharge" in text:
        return "Mobile Recharge"
    if any(x in text for x in ["meesho", "amazon", "flipkart"]):
        return "Shopping"
    if any(x in text for x in ["uber", "ola", "rapido", "auto", "bus", "train"]):
        return "Transport"
    if any(x in text for x in ["salary", "stipend"]):
        return "Salary/Income"
    if "received from" in text:
        return "Personal Transfer"
    return "Personal Transfer"

df["category"] = df["Transaction Details"].apply(classify)

df["subcategory"] = df["category"].map({
    "Food & Dining": "Restaurant",
    "Grocery": "Groceries",
    "Mobile Recharge": "Prepaid Recharge",
    "Shopping": "Online Shopping",
    "Transport": "Auto",
    "Salary/Income": "Received Transfer",
    "Personal Transfer": "Friend/Family"
}).fillna("Friend/Family")

df["expense_nature"] = df.apply(
    lambda x: "Income" if x["transaction_type"] == "CREDIT"
    else "Essential" if x["category"] in ["Grocery", "Mobile Recharge", "Transport"]
    else "Discretionary" if x["category"] in ["Food & Dining", "Shopping"]
    else "Transfer",
    axis=1
)

df["flexibility_level"] = df["expense_nature"].map({
    "Essential": "Low",
    "Discretionary": "High",
    "Transfer": "Not Applicable",
    "Income": "Not Applicable"
})

df["flexibility_score"] = df["flexibility_level"].map({
    "Low": 0.3,
    "Medium": 0.5,
    "High": 0.8,
    "Not Applicable": 0
})

df["amount_band"] = pd.cut(
    df["amount"],
    bins=[-1, 100, 500, 1000, 5000, float("inf")],
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

df["monthly_budget"] = 30000
df["remaining_budget"] = 30000
df["savings_goal"] = 100000

output_columns = [
    "transaction_id", "transaction_date", "transaction_time",
    "transaction_details", "utr", "transaction_type",
    "payment_method", "amount", "currency", "merchant",
    "subcategory", "category", "cashflow_type", "expense_nature",
    "flexibility_level", "flexibility_score", "transaction_status",
    "notes", "city", "is_expense", "amount_band", "day_of_week",
    "month", "year", "monthly_budget", "remaining_budget",
    "savings_goal"
]

df[output_columns].to_csv("data/phonepe_processed.csv", index=False)

print("Processed PhonePe dataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(output_columns))