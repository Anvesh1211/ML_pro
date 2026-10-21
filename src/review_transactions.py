import pandas as pd
import os

RESULT_FILE = "data/final_phonepe_results.csv"
FEEDBACK_FILE = "data/user_feedback.csv"

df = pd.read_csv(RESULT_FILE)

categories = [
    "Bills & Utilities",
    "Education",
    "Entertainment",
    "Food & Dining",
    "Grocery",
    "Health",
    "Mobile Recharge",
    "Personal Care",
    "Personal Transfer",
    "Rent",
    "Salary/Income",
    "Shopping",
    "Transport"
]

if os.path.exists(FEEDBACK_FILE):
    feedback = pd.read_csv(FEEDBACK_FILE)
else:
    feedback = pd.DataFrame(columns=["merchant", "final_category"])

review = df[df["final_category"] == "Needs Review"].copy()

if len(review) == 0:
    print("No transactions need review.")
    exit()

review = review.drop_duplicates(
    subset=["merchant"]
)

print("\n========================================")
print("       FIN TWIN USER REVIEW")
print("========================================")

print(f"\nUnique merchants requiring review: {len(review)}")

for index, row in review.iterrows():

    merchant_name = str(row["merchant"]).strip()

    print("\n----------------------------------------")
    print(f"Merchant : {merchant_name}")
    print(f"Amount   : ₹{row['amount']:.2f}")
    print(f"Type     : {row['transaction_type']}")

    print("\nChoose category:")

    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")

    while True:
        try:
            choice = int(input("Enter category number: "))

            if 1 <= choice <= len(categories):
                selected_category = categories[choice - 1]
                break

            print("Please enter a valid number.")

        except ValueError:
            print("Please enter a number.")

    normalized_merchant = merchant_name.lower()

    same_merchant = (
        df["merchant"]
        .fillna("")
        .str.strip()
        .str.lower()
        == normalized_merchant
    )

    df.loc[same_merchant, "final_category"] = selected_category
    df.loc[same_merchant, "category_source"] = "User"

    existing = feedback[
        feedback["merchant"]
        .fillna("")
        .str.strip()
        .str.lower()
        == normalized_merchant
    ]

    if len(existing) > 0:
        feedback.loc[
            existing.index, "final_category"
        ] = selected_category
    else:
        feedback.loc[len(feedback)] = [
            merchant_name,
            selected_category
        ]

    updated_count = same_merchant.sum()

    print(f"\nSaved: {selected_category}")
    print(f"Updated {updated_count} transaction(s).")
    
df.to_csv(RESULT_FILE, index=False)
df.to_csv("data/phonepe_predictions.csv", index=False)

feedback.to_csv(FEEDBACK_FILE, index=False)

print("\n========================================")
print("       USER FEEDBACK SAVED")
print("========================================")

print(f"\nFeedback records: {len(feedback)}")
print(f"Saved to: {FEEDBACK_FILE}")
print("\nAll corrections have been permanently stored.")