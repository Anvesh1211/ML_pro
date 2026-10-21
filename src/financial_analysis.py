import pandas as pd

df = pd.read_csv("data/final_phonepe_results.csv")

print("\n========================================")
print("       FINANCIAL HEALTH ANALYSIS")
print("========================================")

income = float(input("\nEnter your monthly income: ₹"))
savings_goal = float(input("Enter your monthly savings goal: ₹"))

expenses = df[df["transaction_type"] == "DEBIT"].copy()
income_transactions = df[
    (df["transaction_type"] == "CREDIT") &
    (df["final_category"] == "Salary/Income")
].copy()

total_expenses = expenses["amount"].sum()
recorded_income = income_transactions["amount"].sum()

essential_categories = [
    "Grocery",
    "Bills & Utilities",
    "Health",
    "Transport",
    "Rent",
    "Education",
    "Mobile Recharge"
]

essential_expenses = expenses[
    expenses["final_category"].isin(essential_categories)
]["amount"].sum()

discretionary_expenses = total_expenses - essential_expenses

recorded_surplus = income - total_expenses

required_after_savings = income - savings_goal

if savings_goal <= recorded_surplus:
    savings_status = "Savings goal is achievable"
else:
    savings_status = "Savings goal is difficult with current spending"

if total_expenses == 0:
    expense_ratio = 0
else:
    expense_ratio = (total_expenses / income) * 100

if expense_ratio < 50:
    spending_status = "Low spending"
elif expense_ratio < 70:
    spending_status = "Moderate spending"
elif expense_ratio < 90:
    spending_status = "High spending"
else:
    spending_status = "Very high spending"

if recorded_surplus >= savings_goal:
    health_score = 90
elif recorded_surplus >= savings_goal * 0.75:
    health_score = 75
elif recorded_surplus >= savings_goal * 0.50:
    health_score = 60
elif recorded_surplus >= 0:
    health_score = 45
else:
    health_score = 25

print("\n----------- FINANCIAL SUMMARY -----------")

print(f"Monthly Income: ₹{income:,.2f}")
print(f"Savings Goal: ₹{savings_goal:,.2f}")
print(f"Total Recorded Expenses: ₹{total_expenses:,.2f}")
print(f"Essential Expenses: ₹{essential_expenses:,.2f}")
print(f"Discretionary Expenses: ₹{discretionary_expenses:,.2f}")
print(f"Recorded Surplus: ₹{recorded_surplus:,.2f}")

print("\n----------- SPENDING ANALYSIS -----------")

print(f"Expense Ratio: {expense_ratio:.2f}%")
print(f"Spending Status: {spending_status}")

print("\n----------- SAVINGS ANALYSIS -----------")

print(f"Amount Needed After Savings: ₹{required_after_savings:,.2f}")
print(f"Savings Status: {savings_status}")

print("\n----------- FINANCIAL HEALTH -----------")

print(f"Financial Health Score: {health_score}/100")

if health_score >= 80:
    print("Financial Health: Excellent")
elif health_score >= 65:
    print("Financial Health: Good")
elif health_score >= 50:
    print("Financial Health: Moderate")
elif health_score >= 35:
    print("Financial Health: Needs Improvement")
else:
    print("Financial Health: Poor")

print("\n----------- CATEGORY SPENDING -----------")

category_spending = (
    expenses.groupby("final_category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

print(category_spending.to_string())

print("\n----------- TOP EXPENSES -----------")

top_expenses = expenses.sort_values(
    "amount",
    ascending=False
)[
    ["merchant", "amount", "final_category"]
].head(10)

print(top_expenses.to_string(index=False))

print("\n========================================")
print("Financial analysis completed.")
print("========================================")