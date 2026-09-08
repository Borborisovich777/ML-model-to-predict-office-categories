"""CLI stages: smoke/evaluate, submit, document, check."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (CANDIDATES, CLASSES, FOLDS, HOLDOUT_SIZE, SEED, SELECTION_RULE,
                     SINGLES, TARGET, WEIGHTS, model_parameters)
from .evaluation import (add_ensembles, aligned_probabilities, combine, folds, scores,
                         select_candidate, split_indices)
from .preprocessing import engineer_features, make_pipeline


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    paths = [*sorted((root / "office_ml").glob("*.py")), root / "train_evaluate.py",
             root / "requirements.txt", root / "requirements-lock.txt"]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def versions():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost",
                "lightgbm", "matplotlib", "joblib", "pytest")
    return {"python": sys.version, "platform": platform.platform(),
            **{p: importlib.metadata.version(p) for p in packages}}


def read_csv(path):
    # Literal 'NA' is a category; only empty fields mean missing.
    columns = pd.read_csv(path, nrows=0).columns
    ids = {c: str for c in columns if c.lower() in ("id", "rowid", "row_id")}
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=ids)


def load_training(root):
    frame = read_csv(root / "office_train.csv")
    if TARGET not in frame or frame[TARGET].isna().any():
        raise ValueError("Missing target")
    y = frame[TARGET]
    if set(y.unique()) != set(CLASSES):
        raise ValueError("Expected labels 0 through 4")
    X = frame.drop(columns=TARGET)
    # Exclude explicit identifiers, if introduced in a future source revision.
    identifiers = [c for c in X if c.lower() in ("id", "rowid", "row_id")]
    return frame, X.drop(columns=identifiers), y.astype(int)


def fit_cv_fold(fold, train_idx, valid_idx, X, y, parameters, n_splits):
    probabilities, timing = {}, []
    print(f"Starting CV fold {fold + 1}/{n_splits}", flush=True)
    for name in (*SINGLES, "dummy"):
        start = time.perf_counter()
        pipeline = make_pipeline(name, X.iloc[train_idx], parameters)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities[name] = aligned_probabilities(pipeline, X.iloc[valid_idx])
        elapsed = time.perf_counter() - start
        timing.append(dict(stage="cv", fold=fold, model=name, seconds=elapsed))
        print(f"CV fold {fold + 1}/{n_splits}: {name} finished in {elapsed:.1f}s", flush=True)
    return fold, valid_idx, probabilities, timing


def collect_cv(X, y, output, parameters, n_splits, workers=1):
    oof = {name: np.full((len(y), len(CLASSES)), np.nan) for name in (*SINGLES, "dummy")}
    fold_ids = np.full(len(y), -1, dtype=int)
    timing = []
    tasks = [(fold, a, b, X, y, parameters, n_splits)
             for fold, (a, b) in enumerate(folds(y, n_splits))]

    def consume(finished):
        fold, valid_idx, predictions, times = finished
        fold_ids[valid_idx] = fold
        for name, p in predictions.items():
            oof[name][valid_idx] = p
        timing.extend(times)
        np.savez_compressed(output / f"fold_{fold}_predictions.npz", **predictions,
                            source_rows=X.iloc[valid_idx].index.to_numpy())

    if workers == 1:
        for task in tasks:
            consume(fit_cv_fold(*task))
    else:
        # Spawn avoids inheriting initialized native-library thread pools.
        with ProcessPoolExecutor(max_workers=min(workers, n_splits),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(fit_cv_fold, *task) for task in tasks]
            for future in as_completed(futures):
                consume(future.result())
    oof = add_ensembles(oof)
    cv = {}
    for name in CANDIDATES:
        per_fold = [scores(y.iloc[fold_ids == f], oof[name][fold_ids == f]) for f in range(n_splits)]
        cv[name] = {"oof": scores(y, oof[name]), "folds": per_fold,
                    "mean": {key: float(np.mean([r[key] for r in per_fold])) for key in per_fold[0]},
                    "std": {key: float(np.std([r[key] for r in per_fold], ddof=1)) for key in per_fold[0]}}
    np.savez_compressed(output / "development_oof.npz", **oof, labels=y.to_numpy(),
                        source_rows=X.index.to_numpy(), fold_ids=fold_ids, classes=CLASSES)
    return cv, timing


def fit_complete_partition(name, X, y, parameters):
    start = time.perf_counter()
    pipeline = make_pipeline(name, X, parameters)
    pipeline.fit(X, y)
    elapsed = time.perf_counter() - start
    print(f"Complete-partition refit: {name} finished in {elapsed:.1f}s", flush=True)
    return name, pipeline, elapsed


def fit_models(names, X, y, parameters, workers):
    if workers == 1:
        return [fit_complete_partition(n, X, y, parameters) for n in names]
    with ProcessPoolExecutor(max_workers=min(workers, len(names)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [pool.submit(fit_complete_partition, n, X, y, parameters) for n in names]
        return [future.result() for future in pending]


def evaluate(root, output, smoke=False, workers=1):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty artifact directory; never overwrite an evaluated holdout run")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    frame, X, y = load_training(root)
    dev, holdout = split_indices(y)
    np.savez_compressed(output / "split_indices.npz", development=dev, holdout=holdout)
    if smoke:
        dev, _ = train_test_split(dev, train_size=1000, stratify=y.iloc[dev], random_state=SEED)
    params = model_parameters(smoke)
    manifest = dict(seed=SEED, holdout_size=HOLDOUT_SIZE, folds=2 if smoke else FOLDS,
                    classes=list(CLASSES), parameters=params, weights=WEIGHTS,
                    candidate_order=list(CANDIDATES), selection_rule=SELECTION_RULE,
                    preprocessing="10 row-local engineered features; native CatBoost / median + one-hot",
                    raw_train_sha256=digest(root / "office_train.csv"),
                    source_sha256=source_hashes(root), versions=versions(), workers=workers,
                    mode="smoke" if smoke else "full",
                    started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "protocol.json", manifest)
    cv, timings = collect_cv(X.iloc[dev], y.iloc[dev], output, params, manifest["folds"], workers)
    winner = select_candidate({name: cv[name]["oof"] for name in CANDIDATES})
    frozen = {**manifest, "selected_model": winner, "cv": cv,
              "frozen_before_holdout_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "frozen_selection.json", frozen)
    if smoke:
        write_json(output / "smoke_metrics.json", dict(cv=cv, selected_model=winner,
                   holdout_evaluated=False, elapsed_seconds=time.perf_counter() - start, timings=timings))
        print("Smoke complete: development subset only; no holdout scores or submission.", flush=True)
        return
    print(f"Selection frozen: {winner}. Fitting complete development partition.", flush=True)
    fitted = {}
    for name, pipeline, elapsed in fit_models((*SINGLES, "dummy"), X.iloc[dev], y.iloc[dev], params, workers):
        fitted[name] = pipeline
        timings.append(dict(stage="development_refit", model=name, seconds=elapsed))
    # First holdout transform/predict/score, after all choices are persisted.
    holdout_started = datetime.now(timezone.utc).isoformat()
    probabilities = add_ensembles({name: aligned_probabilities(model, X.iloc[holdout])
                                  for name, model in fitted.items()})
    holdout_scores = {name: scores(y.iloc[holdout], probabilities[name], detailed=True)
                      for name in CANDIDATES}
    np.savez_compressed(output / "holdout_predictions.npz", **probabilities,
                        source_rows=holdout, labels=y.iloc[holdout].to_numpy(), classes=CLASSES)
    engineered = engineer_features(X.iloc[dev])
    metadata = dict(train_shape=list(frame.shape), raw_predictors=X.shape[1],
                    engineered_predictors=engineered.shape[1], development_rows=len(dev),
                    holdout_rows=len(holdout), target_counts={str(c): int((y == c).sum()) for c in CLASSES},
                    numeric_predictors=len(engineered.select_dtypes(include=np.number).columns),
                    categorical_predictors=len(engineered.select_dtypes(exclude=np.number).columns))
    result = dict(dataset=metadata, cv=cv, holdout=holdout_scores, selected_model=winner,
                  selection_rule=SELECTION_RULE, protocol=manifest,
                  frozen_selection_sha256=digest(output / "frozen_selection.json"),
                  holdout_evaluation_started_at=holdout_started,
                  timings=timings, elapsed_seconds=time.perf_counter() - start)
    write_json(output / "metrics.json", result)
    rows = []
    for name in CANDIDATES:
        row = dict(model=name, selected=name == winner)
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            row.update({f"cv_{metric}_mean": cv[name]["mean"][metric],
                        f"cv_{metric}_std": cv[name]["std"][metric],
                        f"oof_{metric}": cv[name]["oof"][metric],
                        f"holdout_{metric}": holdout_scores[name][metric]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)
    from .reporting import plot_confusion_matrices
    plot_confusion_matrices(result, output)
    print(f"Full evaluation complete in {result['elapsed_seconds']:.1f}s; winner remains {winner}.", flush=True)


def submission_identifiers(test, template):
    """Preserve source IDs or use an explicitly supplied, row-aligned template."""
    source_ids = [c for c in test if c.lower() in ("id", "rowid", "row_id")]
    if len(source_ids) > 1:
        raise ValueError("Ambiguous identifier columns")
    if source_ids:
        name = source_ids[0]
        ids = test[name].copy()
        feature_frame = test.drop(columns=name)
        provenance = f"source test column {name}"
    else:
        if template is None or list(template.columns) != ["Id", TARGET] or len(template) != len(test):
            raise ValueError("No test ID: require explicit Id/OfficeCategory template with matching row count")
        name, ids, feature_frame = "Id", template["Id"].copy(), test.copy()
        provenance = "existing repository submission template; official competition format unverified"
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Submission IDs must be nonmissing and unique")
    return name, ids.to_numpy(), feature_frame, provenance


def submit(root, output, template_path, workers=1):
    start = time.perf_counter()
    result = json.loads((output / "metrics.json").read_text())
    frozen = json.loads((output / "frozen_selection.json").read_text())
    if result["protocol"]["mode"] != "full" or "holdout" not in result:
        raise ValueError("Complete evaluation must precede submission")
    if digest(output / "frozen_selection.json") != result["frozen_selection_sha256"]:
        raise ValueError("Frozen selection changed after evaluation")
    if source_hashes(root) != frozen["source_sha256"]:
        raise ValueError("Source changed after freezing; do not refit a different pipeline")
    if versions() != frozen["versions"]:
        raise ValueError("Runtime versions changed after evaluation")
    if digest(root / "office_train.csv") != frozen["raw_train_sha256"]:
        raise ValueError("Labeled data changed after evaluation")
    if (output / "submission.csv").exists():
        raise FileExistsError("Submission already exists; use the saved artifact")
    _, X, y = load_training(root)
    test = read_csv(root / "office_test.csv")
    template = read_csv(template_path) if template_path else None
    id_name, ids, X_test, provenance = submission_identifiers(test, template)
    if set(X_test.columns) != set(X.columns):
        raise ValueError("Test predictors do not match training predictors")
    selected = frozen["selected_model"]
    names = SINGLES if selected in WEIGHTS else (selected,)
    fitted, probabilities = {}, {}
    for name, pipeline, elapsed in fit_models(names, X, y, frozen["parameters"], workers):
        fitted[name] = pipeline
        probabilities[name] = aligned_probabilities(pipeline, X_test)
        print(f"Full-data submission refit complete: {name}", flush=True)
    p = combine(probabilities, frozen["weights"][selected]) if selected in WEIGHTS else probabilities[selected]
    prediction = np.asarray(CLASSES)[p.argmax(axis=1)]
    submission = pd.DataFrame({id_name: ids, TARGET: prediction})
    if len(submission) != len(test) or not np.array_equal(submission[id_name].to_numpy(), ids):
        raise AssertionError("Submission identifier/row contract broken")
    submission.to_csv(output / "submission.csv", index=False)
    joblib.dump(dict(models=fitted, selected_model=selected, classes=CLASSES,
                     weights=frozen["weights"].get(selected)), output / "final_model.joblib", compress=3)
    np.savez_compressed(output / "submission_probabilities.npz", probabilities=p,
                        ids=np.asarray(ids, dtype=str), classes=CLASSES)
    write_json(output / "submission_metadata.json", dict(selected_model=selected, rows=len(test),
               test_shape=list(test.shape), identifier_provenance=provenance,
               test_sha256=digest(root / "office_test.csv"),
               template_sha256=digest(template_path) if template_path else None,
               submission_sha256=digest(output / "submission.csv"),
               model_sha256=digest(output / "final_model.joblib"),
               elapsed_seconds=time.perf_counter() - start, workers=workers,
               labeled_test_evaluation=False))
    print(f"Saved {len(submission)} predictions; unlabeled test accuracy is unavailable.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "submit", "document", "check"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--id-template", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent model/fold processes; each model uses one CPU thread")
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if args.smoke and args.stage != "evaluate":
        parser.error("--smoke applies only to evaluate")
    if args.workers < 1:
        parser.error("--workers must be positive")
    captured = []
    original = warnings.showwarning

    def record_warning(message, category, filename, lineno, file=None, line=None):
        captured.append(dict(message=str(message), category=category.__name__, filename=filename, lineno=lineno))
        original(message, category, filename, lineno, file, line)

    warnings.showwarning = record_warning
    warnings.simplefilter("always")
    completed = False
    try:
        if args.stage == "evaluate":
            evaluate(root, output, args.smoke, args.workers)
        elif args.stage == "submit":
            submit(root, output, args.id_template, args.workers)
        else:
            from .reporting import check_artifacts, write_documentation
            if args.stage == "document":
                write_documentation(root, output)
            else:
                check_artifacts(root, output)
        completed = True
    finally:
        warnings.showwarning = original
        if completed and output.exists():
            write_json(output / f"{args.stage}_warnings.json", captured)


if __name__ == "__main__":
    main()
