"""Independently recompute fold summaries/votes and verify immutable inputs.

Run from the repository root: python verification/audit_saved_predictions.py
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from office_ml.config import CANDIDATES, CLASSES, FOLDS, WEIGHTS
from office_ml.evaluation import combine, folds, scores

root = Path(__file__).resolve().parents[1]
artifact = root / 'artifacts'
result = json.loads((artifact / 'metrics.json').read_text())
initial = json.loads((root / 'legacy/initial_inspection.json').read_text())
for filename in ['nagibator.ipynb', 'AI1010_Group_Assignment_3_Report.pdf',
                 'intro-to-ai-main.zip', 'office_train.csv', 'office_test.csv',
                 'clean.csv', 'submission.csv']:
    assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == initial['files'][filename]
with np.load(artifact / 'split_indices.npz') as split, np.load(artifact / 'development_oof.npz') as oof, np.load(artifact / 'holdout_predictions.npz') as holdout:
    assert set(split['development']).isdisjoint(split['holdout'])
    assert len(set(split['development']) | set(split['holdout'])) == 35000
    np.testing.assert_array_equal(oof['source_rows'], split['development'])
    np.testing.assert_array_equal(holdout['source_rows'], split['holdout'])
    np.testing.assert_array_equal(oof['classes'], CLASSES)
    np.testing.assert_array_equal(holdout['classes'], CLASSES)
    for fold, (_, valid) in enumerate(folds(oof['labels'])):
        np.testing.assert_array_equal(np.flatnonzero(oof['fold_ids'] == fold), valid)
    for saved in [oof, holdout]:
        for ensemble, weights in WEIGHTS.items():
            np.testing.assert_allclose(saved[ensemble], combine(saved, weights), atol=1e-15, rtol=0)
    for name in CANDIDATES:
        computed = [scores(oof['labels'][oof['fold_ids'] == f], oof[name][oof['fold_ids'] == f]) for f in range(FOLDS)]
        assert computed == result['cv'][name]['folds']
        for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            values = [r[key] for r in computed]
            assert np.mean(values) == result['cv'][name]['mean'][key]
            assert np.std(values, ddof=1) == result['cv'][name]['std'][key]
print('PASS: raw files unchanged; split/fold membership, both voting formulas, and every CV mean/SD verified.')
