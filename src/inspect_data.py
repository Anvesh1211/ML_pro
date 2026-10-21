import pandas as pd

training = pd.read_csv("data/training.csv")
testing = pd.read_csv("data/testing.csv", skiprows=2)

print("TRAINING:", training.shape)
print("TESTING:", testing.shape)

print("\nTraining columns:")
print(training.columns.tolist())

print("\nPhonePe columns:")
print(testing.columns.tolist())

print("\nPhonePe transaction types:")
print(testing["Transaction Type"].value_counts())

print("\nPhonePe sample:")
print(testing[["Date", "Transaction Details", "Transaction Type", "Amount"]].head(10))