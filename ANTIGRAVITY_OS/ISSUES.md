# FinTwin AI — Issues and Limitations

## Resolved in the current implementation

- Prediction results are written atomically to both expected output CSVs and
  read back for row/schema validation.
- Training and inference now share the same description + merchant +
  subcategory text construction.
- Saved user feedback is loaded and applied after deterministic rules; the ML
  prediction remains available as a separate field.
- Feedback supports transaction scope and exact normalized merchant scope.
- Seven PhonePe statement footer lines are explicitly audited rather than
  silently treated as data.
- Analysis uses a consistent `final_category` field and separates personal
  transfer debits from consumption expenses.
- Month-wise calculations replace all-month totals compared against monthly
  inputs.
- Hardcoded budget/savings placeholders have been removed from preprocessing.
- Clustering and anomaly logic now consumes the final ledger schema.
- The current virtualenv executable has been verified as runnable; dependency
  versions are pinned.

## Remaining limitations

1. PhonePe final categories are not independently verified labels; no accuracy
   claim is made for those transactions.
2. `user_feedback.csv` has eight pre-existing merchant-wide corrections. The
   old format contains no transaction scope metadata; those rows are preserved
   and interpreted as the merchant-level corrections entered by the earlier
   review tool.
3. The statement spans only a few months and has partial boundary months.
   Long-term seasonality and reliable salary forecasting are unavailable.
4. No account balances, debt obligations, investment balances, or verified
   savings totals are present. A financial-health score is withheld.
5. Expense flexibility is a category-level proxy; a merchant/category cannot
   prove individual purpose or necessity.
6. Recurrence status is descriptive of this statement only. Even a
   three-observation pattern is not a contractual commitment.
7. Isolation Forest has no verified fraud labels. Its flags are exploratory
   review signals, not fraud detections.
8. No adaptive-personalization accuracy claim on PhonePe is possible without
   manually reviewed ground-truth categories. Offline simulation is reported
   separately from user feedback.
9. The app is local Streamlit only; no cloud deployment or authentication is
   configured. Use on a trusted local device and protect the statement and
   local feedback files.
10. Reports currently provide JSON and redacted CSV exports; PDF generation
    and automated visual-regression testing are not part of the verified
    scope. A manual browser check was performed at desktop and 390px widths.
