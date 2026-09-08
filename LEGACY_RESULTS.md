# Superseded historical results

The original notebook, report, ZIP, clean.csv, root submission.csv and catboost_info/ logs are retained. They are historical records, not evidence for the repaired evaluation. Original README.md, results.md, setup guide and submission.csv are also preserved in legacy/.

The old notebook fits medians, categorical modes and target encodings on all labeled rows before splitting. It reuses one validation partition for early stopping, model comparison and final selection, then applies newly fitted full-data preprocessing to a model fitted on a smaller partition. It also sets XGBoost's estimator count from LightGBM's best iteration. These invalidate the old performance claims as independent evaluation.

Do not use the old notebook as the reproducible entry point. Use train_evaluate.py and inspect the repaired artifacts/metrics.json. The root submission.csv is an unchanged historical file; new predictions are artifacts/submission.csv.

Superseded claims include 86.37% validation accuracy, 90.42% training accuracy, 84.5% test accuracy, old per-class scores, feature importances, cleaning counts, improvement attribution, cross-validation and winner claims. No labeled external test file was supplied. The repaired pipeline generates its own results rather than assuming these numbers.

No original dataset, notebook, PDF, ZIP, generated prediction or training log has been deleted or overwritten. Legacy files retain their original claims solely as an explicitly superseded record.
