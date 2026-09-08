# Verification record

The completed full run uses the fixed five-fold development protocol in `artifacts/protocol.json`. Eighteen focused tests passed. Two serial smoke runs and one parallel smoke run produced identical scores, selections, fold membership and every OOF probability. Smoke used development rows only. Bitwise equivalence was tested on smoke paths in this environment, not claimed across hardware or from a duplicate complete run.

Recorded evaluation time through metric generation: 1171.187 seconds (19.52 minutes). Full-data submission refit and artifact generation: 735.905 seconds (12.27 minutes). Evaluation used 5 processes, each estimator using one CPU thread. Per-fit timings are in metrics.json. These figures exclude initial setup, tests, the interrupted development-only attempts, and diagnostics.

Selected model: `ensemble_equal`. The frozen-selection file was persisted before holdout scoring. There were no completed holdout evaluations in either interrupted attempt. The published holdout was evaluated once by each fitted candidate; later audit checks only recompute metrics from saved predictions.

The exact commands, environmental fixes, development-only restarts, and runtime diagnostics are in [verification/commands.md](verification/commands.md). Full logs are in [verification/logs/](verification/logs/).

## Environment

| Component | Version |
|---|---|
| python | 3.11.15 (main, Mar  3 2026, 00:52:57) [Clang 17.0.0 (clang-1700.6.3.2)] |
| platform | macOS-26.6.2-arm64-arm-64bit |
| numpy | 2.2.6 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| catboost | 1.2.8 |
| xgboost | 3.0.2 |
| lightgbm | 4.6.0 |
| matplotlib | 3.10.3 |
| joblib | 1.5.1 |
| pytest | 8.3.5 |

System OpenMP: Homebrew libomp 23.1.0. All Python dependencies were installed in a task-local virtual environment; the complete Python package set is pinned in requirements-lock.txt.

## Warnings and fixes

Final unit tests reported 13 Matplotlib/Pyparsing deprecation warnings. They do not concern training data, targets or numerical validity. A LightGBM feature-name warning found during development was fixed by returning consistently named feature matrices. Structured coordinator warning counts across completed stages: {'PyparsingDeprecationWarning': 27}. Worker warnings, when emitted, are in the combined stage logs. Warnings were retained rather than presented as accuracy evidence.

Initial sandbox DNS failures were resolved through approved network access. The missing OpenMP runtime was installed. A documentation table-format fix and independent-process scheduling caused two development-only restarts, documented in commands.md. Neither changed model hyperparameters or used holdout scores.

## Final checks

- `python train_evaluate.py check --output-dir artifacts`: stored metrics, probability contracts, frozen selection, submission checksum and generated documentation agree.
- `python verification/audit_saved_predictions.py`: independently recomputed each CV fold summary and voting formula, checked split membership and historical-file checksums.
- `python verification/check_smoke.py`: serial/repeated/parallel smoke predictions match exactly.
- `python verification/check_saved_model.py`: reloaded final pipelines reproduce the first 100 saved unlabeled-test probability rows.
- `git diff --check`: no whitespace errors.
- At the end of the evaluation phase, Git HEAD was the initial commit and no remote state had been changed. Publication was authorized separately afterward; the benchmark source and numerical artifacts remain unchanged.

The actual command logs and final status are retained in verification/logs. External-test accuracy and official competition submission acceptance cannot be checked because labeled external data and the official template were not provided.
