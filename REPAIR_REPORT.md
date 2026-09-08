# Office ML evaluation repair

The evaluation was completed locally before publication was separately authorized. This change publishes that verified repair for review; model settings and measured results are unchanged.

## Root causes and repair

The notebook fitted imputation and target encodings before splitting, reused one validation set for early stopping/model selection, copied LightGBM's iteration count into XGBoost, and refitted preprocessing inconsistently for submission. Historical documentation made unsupported CV, performance, cleaning and feature-importance claims.

The replacement splits raw rows first, fits preprocessing within each fold, obtains aligned OOF probabilities for all candidates, compares two fixed voting weight sets, and saves the selection before holdout scoring. Independently specified iteration counts are 300/240/320 for CatBoost/XGBoost/LightGBM. No early stopping or weight search is used. The selected pipeline is then refitted on all 35,000 labeled rows and produces 15,000 submission predictions using preserved template IDs.

## Files changed or added

- `train_evaluate.py` and `office_ml/`: configuration, isolated preprocessing, model factories, CV/holdout stages, frozen selection, submission refitting and documentation generation.
- `tests/test_pipeline.py`: 18 tests for fold isolation, holdout guards, unseen categories, finite inputs, actual model determinism, class/probability alignment, serializable fitted pipelines and identifiers.
- `requirements.txt`, `requirements-lock.txt`, `.gitignore`: pinned environment and local cache handling.
- `README.md`, `results.md`, `GITHUB_SETUP_AND_RUN.md`, `LEGACY_RESULTS.md`: replaced unsupported current claims and identified historical material.
- `legacy/`: byte-preserved original documentation and submission, plus initial inspection hashes.
- `artifacts/`: metrics JSON, numeric comparison CSV, complete per-class reports/confusion matrices, raw row partitions, OOF and holdout probabilities, frozen protocol, final fitted model and new submission.
- `artifacts-smoke*`, `verification/`, `VERIFICATION.md`: smoke reproducibility evidence, command logs, runtime, warning and audit records.

The original notebook, PDF, ZIP, raw datasets, clean.csv, root submission.csv and CatBoost logs are preserved.

## Repaired results

| Model | CV accuracy ± SD | Holdout accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| catboost | 84.43% ± 0.75 pp | 85.17% | 0.8524 | 85.22% |
| xgboost | 83.81% ± 0.76 pp | 84.49% | 0.8454 | 84.55% |
| lightgbm | 84.71% ± 0.85 pp | 85.51% | 0.8556 | 85.57% |
| ensemble_equal (selected) | 85.08% ± 0.77 pp | 85.77% | 0.8582 | 85.82% |
| ensemble_weighted | 85.06% ± 0.77 pp | 85.86% | 0.8591 | 85.91% |
| dummy | 19.93% ± 0.72 pp | 19.86% | 0.1985 | 19.85% |

CV is development-only; SD is across five folds, in percentage points. Macro F1 in this table is holdout macro F1. Full CV macro F1/balanced accuracy, per-class precision/recall/F1/support and confusion matrices are in results.md and artifacts/metrics.json.

Final selection: **ensemble_equal**, by the highest development OOF accuracy, then macro F1, then a fixed tie order. The holdout did not select the final model. README.md gives the numerical ensemble-versus-best-single comparison.

## Commands and verification

From an activated Python 3.11 environment at the repository root:

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
python train_evaluate.py evaluate --smoke --output-dir artifacts-smoke
python train_evaluate.py evaluate --smoke --output-dir artifacts-smoke-repeat
python train_evaluate.py evaluate --smoke --output-dir artifacts-smoke-parallel --workers 2
python verification/check_smoke.py
python train_evaluate.py evaluate --output-dir artifacts --workers 5
python train_evaluate.py submit --output-dir artifacts --id-template legacy/submission.csv --workers 5
python train_evaluate.py document --output-dir artifacts
python train_evaluate.py check --output-dir artifacts
python verification/audit_saved_predictions.py
python verification/check_saved_model.py
git diff --check
```

These are portable equivalents of the exact local interpreter paths in verification/commands.md. Use new artifact-directory names for re-execution because delivered outputs are protected against overwrite. All 18 tests, probability/data audits, document consistency checks and smoke comparisons passed. See VERIFICATION.md for measured runtime, versions and retained warnings.

## Corrected claims and limitations

Old 86.37% validation and 84.5% test figures, training/per-class scores, improvement/importance assertions, cleaning counts, leakage-free claims and old winner claims no longer describe the current results. Historical copies are explicitly superseded.

This repair isolates the holdout within the new run, but seed 42 recreates historically used validation membership and the data informed prior work. It is not a new external evaluation. Development selection adds CV optimism; there is only one holdout split and no significance test. Entity independence and deployment generalization are unverified. External test labels and the official submission template are unavailable, so no external-test accuracy or competition acceptance is claimed.

The historical README names Nurtore Arynuruly, Shakhnazar Sailaukan and Ivan Kanev, but individual original implementation boundaries are unverified. This repair was produced with Codex at Nurtore's request; the wording below does not attribute teammates' work to him.

## Recruiter-safe wording

Project description: Team office-category classification project comparing three boosting models and two soft-voting ensembles with a repaired evaluation; the development-selected equal-weight CatBoost/XGBoost/LightGBM soft-voting ensemble achieved 85.77% accuracy on a 7,000-row holdout within the supplied dataset.

CV bullet: Used Codex to repair evaluation leakage in a team office-category classification project, add fold-fitted preprocessing and stratified five-fold comparison, and obtain 85.77% holdout accuracy (macro F1 0.858) with the development-selected equal-weight CatBoost/XGBoost/LightGBM soft-voting ensemble.
