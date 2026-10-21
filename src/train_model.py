import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/training.csv")

df = df.dropna(subset=["category"])

df["text"] = (
    df["transaction_details"].fillna("") + " " +
    df["merchant"].fillna("") + " " +
    df["subcategory"].fillna("")
)

X = df["text"]
y = df["category"]

groups = df["merchant"].fillna("UNKNOWN_MERCHANT")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

features = FeatureUnion([

    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),

    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True
        )
    )
])

model = Pipeline([

    ("features", features),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print("\n========================================")
print("       FIN TWIN ML MODEL")
print("========================================")

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

print(f"Unique training merchants: {groups.iloc[train_idx].nunique()}")
print(f"Unique testing merchants: {groups.iloc[test_idx].nunique()}")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


joblib.dump(model, "src/category_model.pkl")

print("\nModel saved successfully:")
print("src/category_model.pkl")