"""Render documentation strictly from saved repaired-run artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANDIDATES, CLASSES, SINGLES, WEIGHTS
from .evaluation import scores, select_candidate, validate_probabilities


def plot_confusion_matrices(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for name, ax in zip(CANDIDATES, axes.flat):
        ConfusionMatrixDisplay(np.asarray(result["holdout"][name]["confusion_matrix"]),
                               display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
        suffix = " (selected by development CV)" if name == result["selected_model"] else ""
        ax.set_title(name + suffix, fontsize=10)
    fig.suptitle("OfficeCategory — final holdout confusion matrices", fontsize=16)
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def comparison_table(result):
    lines = ["| Model | CV accuracy ± SD | CV macro F1 ± SD | CV balanced accuracy ± SD | Holdout accuracy | Holdout macro F1 | Holdout balanced accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CANDIDATES:
        cv, h = result["cv"][name], result["holdout"][name]
        label = f"**{name} (selected)**" if name == result["selected_model"] else name
        cells = [f"{100*cv['mean'][m]:.2f}% ± {100*cv['std'][m]:.2f}" for m in
                 ("accuracy", "macro_f1", "balanced_accuracy")]
        cells += [f"{100*h[m]:.2f}%" for m in ("accuracy", "macro_f1", "balanced_accuracy")]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def document_text(result, submission, relative_artifacts):
    data, selected = result["dataset"], result["selected_model"]
    best_single_cv = max(SINGLES, key=lambda n: result["cv"][n]["oof"]["accuracy"])
    best_ensemble_cv = max(WEIGHTS, key=lambda n: result["cv"][n]["oof"]["accuracy"])
    best_single_h = max(SINGLES, key=lambda n: result["holdout"][n]["accuracy"])
    best_ensemble_h = max(WEIGHTS, key=lambda n: result["holdout"][n]["accuracy"])
    cv_delta = 100*(result["cv"][best_ensemble_cv]["oof"]["accuracy"] - result["cv"][best_single_cv]["oof"]["accuracy"])
    h_delta = 100*(result["holdout"][best_ensemble_h]["accuracy"] - result["holdout"][best_single_h]["accuracy"])
    conclusion = (f"Development OOF: strongest ensemble `{best_ensemble_cv}` minus strongest single "
                  f"`{best_single_cv}` = {cv_delta:+.3f} percentage points. "
                  f"Holdout: strongest ensemble `{best_ensemble_h}` minus strongest single "
                  f"`{best_single_h}` = {h_delta:+.3f} percentage points. "
                  f"{'The ensemble beats' if h_delta > 0 else 'The ensemble does not beat'} the strongest single model on holdout accuracy. "
                  "These are descriptive comparisons, not significance tests. The holdout does not change selection.")
    table = comparison_table(result)
    common = f"""Generated from `{relative_artifacts}/metrics.json` and `submission_metadata.json`.

Selected model: **{selected}**, using {result['selection_rule']}.

{table}

CV values summarize five development folds; SD is the sample standard deviation (ddof=1), in percentage points, not a confidence interval. OOF selection scores use all development predictions together. Holdout values are separate, after selection was frozen.

{conclusion}
"""
    readme = f"""# Office building category classification

Reproducible comparison of CatBoost, XGBoost, LightGBM and probability-voting ensembles for the five labels of `OfficeCategory` (0–4). This repaired pipeline supersedes the historical notebook evaluation.

## Verified data

`office_train.csv`: {data['train_shape'][0]:,} rows × {data['train_shape'][1]} columns ({data['raw_predictors']} predictors plus target). `office_test.csv`: {submission['test_shape'][0]:,} rows × {submission['test_shape'][1]} predictors, without labels or a source ID. Class counts: {data['target_counts']}. The classes are approximately balanced, not exactly equal. The repository does not independently establish the provenance or meaning of each quality tier.

## Repaired evaluation

Raw labeled rows are split first, stratified 80/20 with seed 42: {data['development_rows']:,} development and {data['holdout_rows']:,} final holdout. Stratified five-fold CV runs only on development. Every fold creates fresh model/preprocessing pipelines. The same folds are used for every candidate and a stratified DummyClassifier baseline.

All 79 raw predictors are retained. Ten deterministic row-local features describe quality × office space, space/plot, space/restroom, space/meeting room, total area, office share, basement share, construction age, renovation age and recent renovation. Dates use YearListed. Undefined ratios/overflows become missing; finite numbers are capped at ±1e15 before fitting. This yields {data['engineered_predictors']} predictors: {data['numeric_predictors']} numeric and {data['categorical_predictors']} categorical. No feature search or feature-effect claims are made.

CatBoost receives native categorical strings (including unseen categories), a distinct missing category token, and numeric medians fitted on its training partition. Its internal categorical statistics see only training labels. XGBoost and LightGBM use training-fitted numeric median imputation and OneHotEncoder(handle_unknown='ignore'); missing categories have a distinct token. All-missing numeric columns fall back to zero. No external target encoding or scaling is used. Literal `NA` strings are retained as categories; empty CSV fields are missing. `clean.csv` is never read.

Prospective independent iteration budgets are CatBoost 300 (depth 6, learning rate .08), XGBoost 240 (depth 5, rate .06), and LightGBM 320 (31 leaves, rate .04). There is no early stopping, parameter search or use of holdout results to choose counts. These are defensible fixed baselines, not claimed optima. Full parameters and one-thread CPU settings are in `office_ml/config.py` and the saved protocol. `--workers` schedules independent folds/refits in separate processes; the default is one. This run used {result['protocol']['workers']} workers. Serial and parallel smoke predictions were verified identical.

Class probabilities are aligned to [0,1,2,3,4]. The historical weighted hypothesis uses `(1.2*p_catboost + p_xgboost + p_lightgbm)/3.2`: 37.5%, 31.25%, 31.25%. Equal voting uses one third each. Prediction is argmax of the averaged probabilities. Voting does not pick a different best model at each batch or iteration. Only these two weight sets are compared using development OOF predictions.

All decisions are persisted in `frozen_selection.json` before each candidate is fitted on complete development and scored once on the holdout. The selected candidate is then refitted on all labeled rows for submission, even if another candidate has higher holdout accuracy.

## Results

{common}

## Exact reproduction

Python 3.11 is tested. Create a virtual environment, install the locked packages, and run from this repository root. On macOS the boosting wheels may require an OpenMP runtime (`libomp`); see `VERIFICATION.md` for this machine's setup.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python train_evaluate.py evaluate --smoke --output-dir artifacts-smoke
python train_evaluate.py evaluate --output-dir artifacts --workers 5
python train_evaluate.py submit --output-dir artifacts --id-template legacy/submission.csv --workers 5
python train_evaluate.py document --output-dir artifacts
python train_evaluate.py check --output-dir artifacts
```

Evaluation requires a new empty output directory so prior runs cannot be silently overwritten. If the delivered directories already exist, use new names consistently. A smoke run uses only a fixed 1,000-row development subset and two folds of eight iterations; it never scores the holdout and cannot generate a submission. Re-running full evaluation is reproduction of an already reported split, not a fresh independent test.

## Artifacts

`{relative_artifacts}/metrics.json` includes all CV fold scores, holdout metrics, per-class precision/recall/F1/support, confusion matrices, timing and versions. `model_comparison.csv` contains the numeric comparison; `split_indices.npz`, `development_oof.npz` and `holdout_predictions.npz` make it auditable. The configuration and source/data hashes are recorded before holdout evaluation. `final_model.joblib` contains the fitted full-data winning pipelines; load only trusted local model files.

![Holdout confusion matrices]({relative_artifacts}/confusion_matrices.png)

New predictions are `{relative_artifacts}/submission.csv` ({submission['rows']:,} rows). Identifier provenance: {submission['identifier_provenance']}. The existing IDs are preserved in order, rather than silently recreated; the original submission is retained. No official competition template was supplied, so submission-format compliance is provisional. Unlabeled predictions supply no test accuracy evidence.

## Limitations and historical results

The holdout is isolated from all modeling decisions in this repaired execution. However, seed 42 reproduces the old notebook's validation row membership, and the entire dataset informed historical work. This is not a never-before-seen external evaluation. CV was used to choose among candidates and therefore has selection optimism. There is one holdout split, no uncertainty/significance analysis, no labeled external test set and no group/time-aware split. Scores describe these supplied rows; entity independence and deployment generalization are unverified.

The old 86.37% validation claim, 84.5% test claim, claimed improvements, feature rankings, cleaning counts and previous winner claims are not carried forward as evidence. See `LEGACY_RESULTS.md`. The notebook, report, ZIP, `clean.csv`, old submission and CatBoost logs are preserved as historical material; use `train_evaluate.py` to reproduce the repair.

## Team and attribution

The historical README names Shakhnazar Sailaukan (sailaukan), Nurtore Arynuruly (lourinser), and Ivan Kanev (vizior). It does not establish individual implementation boundaries. This repair was prepared with Codex at Nurtore's request; it does not establish that Nurtore alone authored the original models or teammates' work.
"""
    details = ["# Repaired evaluation results", "", common, "", "## Per-class holdout metrics", ""]
    for name in CANDIDATES:
        details += [f"### {name}", "", "| Class | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for c in CLASSES:
            r = result["holdout"][name]["classification_report"][str(c)]
            details.append(f"| {c} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
        details += ["", "Confusion matrix: rows = true labels, columns = predicted labels, ordered 0–4.",
                    "```text", *[" ".join(map(str, row)) for row in result['holdout'][name]['confusion_matrix']], "```", ""]
    details += ["No labeled external-test score is available. The selected model was refitted on all labeled rows only after evaluation. The repaired holdout uses historically seen data; see README limitations."]
    return readme, "\n".join(details) + "\n"


def write_documentation(root, output):
    result = json.loads((output / "metrics.json").read_text())
    submission = json.loads((output / "submission_metadata.json").read_text())
    rel = Path(__import__('os').path.relpath(output, root)).as_posix()
    readme, details = document_text(result, submission, rel)
    (root / "README.md").write_text(readme)
    (root / "results.md").write_text(details)
    (output / "comparison.md").write_text(comparison_table(result) + "\n")
    print("README.md and results.md generated from completed repaired artifacts.")


def check_artifacts(root, output):
    from .train import digest, source_hashes, read_csv
    result = json.loads((output / "metrics.json").read_text())
    frozen = json.loads((output / "frozen_selection.json").read_text())
    sub = json.loads((output / "submission_metadata.json").read_text())
    assert result['frozen_selection_sha256'] == digest(output / 'frozen_selection.json')
    assert frozen['source_sha256'] == source_hashes(root)
    assert frozen['frozen_before_holdout_at'] < result['holdout_evaluation_started_at']
    assert result['selected_model'] == select_candidate({n: result['cv'][n]['oof'] for n in CANDIDATES})
    assert result['selected_model'] == frozen['selected_model'] == sub['selected_model']
    table = pd.read_csv(output / 'model_comparison.csv').set_index('model')
    with np.load(output / 'holdout_predictions.npz') as saved:
        for name in CANDIDATES:
            recalc = scores(saved['labels'], saved[name], detailed=True)
            assert recalc == result['holdout'][name]
            for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
                assert np.isclose(table.loc[name, 'holdout_' + key], recalc[key], atol=1e-12)
                for stat in ('mean', 'std'):
                    assert np.isclose(table.loc[name, f'cv_{key}_{stat}'], result['cv'][name][stat][key], atol=1e-12)
    with np.load(output / 'development_oof.npz') as oof:
        for name in CANDIDATES:
            assert scores(oof['labels'], oof[name]) == result['cv'][name]['oof']
    with np.load(output / 'submission_probabilities.npz') as probs:
        validate_probabilities(probs['probabilities'])
        submission = read_csv(output / 'submission.csv')
        assert len(submission) == sub['rows']
        assert np.array_equal(submission.iloc[:, 0].to_numpy(), probs['ids'])
        assert np.array_equal(submission.iloc[:, 1].to_numpy(), np.asarray(CLASSES)[probs['probabilities'].argmax(axis=1)])
    assert digest(output / 'submission.csv') == sub['submission_sha256']
    rel = Path(__import__('os').path.relpath(output, root)).as_posix()
    readme, details = document_text(result, sub, rel)
    assert (root / 'README.md').read_text() == readme
    assert (root / 'results.md').read_text() == details
    assert (output / 'comparison.md').read_text() == comparison_table(result) + '\n'
    print('Artifact probabilities, metrics, frozen selection, submission and documentation agree.')
