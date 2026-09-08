"""Probability contracts, deterministic partitions and development selection."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import CANDIDATES, CLASSES, FOLDS, HOLDOUT_SIZE, SEED, SINGLES, WEIGHTS


def split_indices(y):
    return train_test_split(np.arange(len(y)), test_size=HOLDOUT_SIZE,
                            stratify=y, random_state=SEED)


def folds(y, n_splits=FOLDS):
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                .split(np.zeros(len(y)), y))


def validate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(CLASSES):
        raise ValueError("Expected one probability column per declared class")
    if not np.isfinite(p).all() or (p < -1e-12).any() or (p > 1 + 1e-12).any():
        raise ValueError("Invalid probability values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows do not sum to one")
    return p


def aligned_probabilities(estimator, X):
    actual = list(np.asarray(estimator.classes_).ravel())
    if len(actual) != len(CLASSES) or set(actual) != set(CLASSES):
        raise ValueError(f"Model class mismatch: {actual}")
    raw = np.asarray(estimator.predict_proba(X))
    return validate_probabilities(raw[:, [actual.index(c) for c in CLASSES]])


def combine(probabilities, weights):
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(SINGLES),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Expected three finite nonnegative weights with positive sum")
    stack = np.stack([validate_probabilities(probabilities[name]) for name in SINGLES])
    return validate_probabilities(np.average(stack, axis=0, weights=w))


def add_ensembles(probabilities):
    return {**probabilities, **{name: combine(probabilities, w) for name, w in WEIGHTS.items()}}


def scores(y, probabilities, detailed=False):
    pred = np.asarray(CLASSES)[validate_probabilities(probabilities).argmax(axis=1)]
    result = dict(accuracy=float(accuracy_score(y, pred)),
                  macro_f1=float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)))
    if detailed:
        result["classification_report"] = classification_report(
            y, pred, labels=CLASSES, output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(y, pred, labels=CLASSES).tolist()
    return result


def select_candidate(development_scores):
    return max(CANDIDATES, key=lambda name: (development_scores[name]["accuracy"],
                development_scores[name]["macro_f1"], -CANDIDATES.index(name)))
