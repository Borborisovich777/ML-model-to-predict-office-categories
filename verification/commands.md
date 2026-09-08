# Executed commands

Working directory for project commands: `outputs/ML-model-to-predict-office-categories`, under this task's directory. `../../work/env/.venv/bin/python` is the isolated Python 3.11.15 interpreter. No remote Git state was changed during the evaluation phase; publication was separately authorized afterward.

## Inspection and environment

```bash
git clone https://github.com/Borborisovich777/ML-model-to-predict-office-categories.git outputs/ML-model-to-predict-office-categories
git status --short
git rev-parse HEAD
git ls-files
python3 --version
/opt/homebrew/bin/python3.11 --version
uname -m
UV_CACHE_DIR="$PWD/work/uv-cache" uv venv --python /opt/homebrew/bin/python3.11 work/env/.venv
UV_CACHE_DIR="$PWD/work/uv-cache" uv pip install --python work/env/.venv/bin/python -r outputs/ML-model-to-predict-office-categories/requirements.txt
HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_ENV_HINTS=1 brew install libomp
UV_CACHE_DIR="$PWD/work/uv-cache" uv pip freeze --python work/env/.venv/bin/python > outputs/ML-model-to-predict-office-categories/requirements-lock.txt
```

The first clone and dependency installation were blocked by sandbox DNS restrictions and were repeated successfully with approved network access. The first uv attempt used its default cache, which was not writable; the task-local `UV_CACHE_DIR` resolved this. XGBoost import then identified missing OpenMP. Homebrew installed libomp 23.1.0. No pre-existing user Python environment was modified; the OpenMP dependency is installed under Homebrew.

## Verification stages, from the repository root

```bash
../../work/env/.venv/bin/python -m pytest -q
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py evaluate --smoke --output-dir artifacts-smoke
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py evaluate --smoke --output-dir artifacts-smoke-repeat
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py evaluate --output-dir artifacts
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py submit --output-dir artifacts --id-template legacy/submission.csv --workers 5
../../work/env/.venv/bin/python train_evaluate.py document --output-dir artifacts
../../work/env/.venv/bin/python train_evaluate.py check --output-dir artifacts
git diff --check
git status --short
```

Shell output was redirected to stage-specific logs, copied into `verification/logs/`. Smoke determinism was checked by exact equality of both JSON CV dictionaries, selected candidate, and every array in both `development_oof.npz` files. The comparison script is `verification/check_smoke.py`.

During testing, returning named numeric matrices removed a LightGBM warning. One assertion needed adjustment for the DataFrame return type; the final suite passed after that fix. An initial full evaluation was interrupted during its first development training fit to correct Markdown table formatting. It produced no OOF results, selection or holdout scores. Its log is retained as `evaluate-interrupted.log`; the published run started afresh with unchanged model parameters.

The historical files were checked by SHA-256 against the initial inspection. Detailed versions, source hashes, dataset hash, timestamps and per-fit runtimes are recorded in `artifacts/protocol.json` and `metrics.json`. Raw warning output and structured stage warning JSON are retained.

## Runtime diagnosis (development data only)

A read-only `sample` of the active training process showed CatBoost computing tree split scores. An eight-iteration timing probe fitted the existing smoke CatBoost parameters on the first full development training fold, without holdout rows or scoring. It took 6.335 seconds; it did not change the selected parameters. The probe log is included for runtime transparency. The model configuration was not changed in response to the probe.

## Parallel scheduling validation

The machine reports 14 CPU cores and 38,654,705,664 bytes of RAM (36 GiB). The serial full-data CatBoost fit remained CPU-active on one core. It was interrupted before its first fold completed and before any holdout use, solely to add independent fold/model process scheduling. Every estimator still uses one CPU thread; model seeds, features, parameters, folds and selection rule are unchanged. The final suite has 18 passing tests, including serialization of fitted pipelines from spawned processes.

The following smoke run was compared against the original serial smoke with exact equality for all saved arrays and CV scores:

```bash
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py evaluate --smoke --output-dir artifacts-smoke-parallel --workers 2
```

The final full evaluation and submission commands use `--workers 5`:

```bash
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py evaluate --output-dir artifacts --workers 5
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python -u train_evaluate.py submit --output-dir artifacts --id-template legacy/submission.csv --workers 5
```

Both interrupted full attempts were development-only and produced no OOF model comparison or holdout scores. Only the completed run in artifacts/ supplies published evaluation evidence. Worker warnings that are emitted are retained in the combined stage log; structured warning JSON captures the coordinator process.

A second development-only eight-iteration diagnostic recorded CatBoost's effective library defaults; it was not a candidate in the reported model comparison. No holdout rows or performance were used. The parallel run was observed using approximately five CPU cores and 7.18 GiB total resident memory, within the available 36 GiB.

## Completed final audits

```bash
../../work/env/.venv/bin/python train_evaluate.py document --output-dir artifacts
../../work/env/.venv/bin/python train_evaluate.py check --output-dir artifacts
../../work/env/.venv/bin/python verification/audit_saved_predictions.py
MPLCONFIGDIR="$PWD/../../work/mplconfig" ../../work/env/.venv/bin/python verification/check_saved_model.py
git diff --check
git diff --quiet HEAD -- nagibator.ipynb AI1010_Group_Assignment_3_Report.pdf intro-to-ai-main.zip office_train.csv office_test.csv clean.csv submission.csv catboost_info
git status --short
git rev-parse HEAD
```

All passed. The confusion-matrix PNG was visually inspected: all six panels, class labels and counts are legible. Git HEAD at the end of evaluation was b252465879331b47d043d5985c911ad91ebf25ea. The 15,000-row new submission and all fitted model objects are saved under artifacts/; the original root submission remains unchanged.

## Publication preparation

Publication was authorized after the complete evaluation and submission run. Only the reports describing publication status and local paths in diagnostic output were adjusted for sharing. `<venv>` and `<task-root>` replace machine-specific paths in published logs and warning records; raw local copies were retained in the task workspace. Benchmark source files, frozen selection, metrics and numerical/model artifacts were not changed. `final-git-status.txt` and `final-head.txt` record the historical end-of-evaluation state, before publication.
