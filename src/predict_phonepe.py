import pandas as pd
import joblib
import re

model = joblib.load("src/category_model.pkl")
df = pd.read_csv("data/phonepe_processed.csv")

df["text"] = (
    df["transaction_details"].fillna("") + " " +
    df["merchant"].fillna("")
)

df["ml_category"] = model.predict(df["text"])

probabilities = model.predict_proba(df["text"])
df["ml_confidence"] = probabilities.max(axis=1)

df["final_category"] = df["ml_category"]
df["category_source"] = "ML"

text = df["text"].str.lower()
merchant = df["merchant"].fillna("").str.lower().str.strip()

rules = {
    "Transport": [
        "petrol", "fuel", "petrol pump", "diesel",
        "irctc", "railway", "redbus", "uber", "ola",
        "rapido", "metro", "taxi", "service station"
    ],

    "Food & Dining": [
        "swiggy", "zomato", "dominos", "pizza",
        "burger", "restaurant", "hotel", "cafe",
        "coffee", "dining", "food", "biryani",
        "mess", "canteen", "bakery", "juice",
        "snack", "pani puri", "vada pav"
    ],

    "Grocery": [
        "grocery", "super market", "supermarket",
        "mart", "market", "kirana", "fruit",
        "vegetable", "vegetables", "traders",
        "grocer", "daily essentials", "blinkit", "zepto"
    ],

    "Shopping": [
        "meesho", "amazon", "flipkart", "myntra",
        "ajio", "shopping", "electronics",
        "clothing", "fashion"
    ],

    "Mobile Recharge": [
        "recharge", "mobile recharge", "prepaid",
        "postpaid"
    ],

    "Bills & Utilities": [
        "electricity", "electric", "gas bill",
        "water bill", "internet", "wifi",
        "broadband", "dth", "utility",
        "bill payment"
    ],

    "Health": [
        "hospital", "medical", "medicine",
        "pharmacy", "chemist", "clinic",
        "doctor", "apollo", "health"
    ],

    "Personal Care": [
        "salon", "unisex salon", "parlour", "beauty",
        "spa", "barber", "haircut"
    ],

    "Education": [
        "college", "school", "education",
        "course", "exam", "tuition", "fees"
    ],

    "Entertainment": [
        "movie", "cinema", "bookmyshow",
        "netflix", "spotify", "entertainment",
        "game", "gaming"
    ],

    "Rent": [
        "rent", "house rent", "room rent"
    ],

    "Salary/Income": [
        "salary", "stipend"
    ],

    "Personal Transfer": [
        "friend transfer",
        "family transfer",
        "received from"
    ]
}

rule_applied = pd.Series(False, index=df.index)

for category, keywords in rules.items():
    pattern = "|".join(
        keyword.replace(" ", r"\s+")
        for keyword in keywords
    )

    mask = (
        text.str.contains(pattern, regex=True, na=False)
        & ~rule_applied
    )

    df.loc[mask, "final_category"] = category
    df.loc[mask, "category_source"] = "Indian Rule"

    rule_applied = rule_applied | mask


business_words = [
    "market", "mart", "store", "shop", "supermarket",
    "grocery", "traders", "foods", "food", "hotel",
    "restaurant", "cafe", "bakery", "hospital",
    "medical", "pharmacy", "clinic", "salon",
    "service", "station", "petrol", "pump", "fuel",
    "rail", "railway", "irctc", "bus", "travel",
    "travels", "recharge", "internet", "school",
    "college", "university", "electronics", "meesho",
    "amazon", "flipkart", "zepto", "blinkit",
    "swiggy", "zomato"
]

def looks_like_person(name):
    words = re.findall(r"[a-z]+", name.lower())

    if len(words) < 2 or len(words) > 4:
        return False

    if any(word in business_words for word in words):
        return False

    return all(len(word) >= 2 for word in words)


person_mask = (
    (df["transaction_type"] == "DEBIT") &
    (df["category_source"] == "ML") &
    merchant.apply(looks_like_person)
)

df.loc[person_mask, "final_category"] = "Personal Transfer"
df.loc[person_mask, "category_source"] = "Person Name Rule"

low_confidence = (
    (df["ml_confidence"] < 0.30) &
    (df["category_source"] == "ML")
)

df.loc[low_confidence, "final_category"] = "Needs Review"
df.loc[low_confidence, "category_source"] = "Low ML Confidence"

manual_rules = {
    "THE KOHINOOR UNISEX SALON": "Personal Care",
    "Priyas South Indian dosa centre": "Food & Dining",
    "jay ganesh general store": "Grocery",
    "LUKDE AND BROTHER S": "Grocery"
}

for merchant_name, category in manual_rules.items():
    mask = merchant.str.lower() == merchant_name.lower()
    df.loc[mask, "final_category"] = category
    df.loc[mask, "category_source"] = "Merchant Rule"

person_transactions = [
    "PUKHARAJ A CHAUDHARI",
    "Mr UZAIR KHAN  ZUBAIR KHAN",
    "SANJOG",
    "Bondya"
]

for merchant_name in person_transactions:
    mask = merchant.str.lower() == merchant_name.lower()
    df.loc[mask, "final_category"] = "Personal Transfer"
    df.loc[mask, "category_source"] = "Merchant Rule"
    
print("\n========================================")
print("       FIN TWIN CLASSIFICATION")
print("========================================")

print("\nTotal transactions:", len(df))

print("\nClassification sources:")
print(df["category_source"].value_counts())

print("\nFinal Categories:")
print(df["final_category"].value_counts())

print("\n===== SAMPLE CLASSIFICATIONS =====")
print(
    df[
        ["merchant", "amount", "transaction_type",
         "final_category", "category_source", "ml_confidence"]
    ].head(20).to_string(index=False)
)

print("\nFiles saved successfully.")