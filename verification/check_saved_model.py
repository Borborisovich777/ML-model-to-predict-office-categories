"""Check the trusted locally generated model round trip, without labeled scoring."""
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from office_ml.evaluation import aligned_probabilities, combine
from office_ml.train import read_csv, submission_identifiers

artifact = root / 'artifacts'
metadata = json.loads((artifact / 'submission_metadata.json').read_text())
assert hashlib.sha256((artifact / 'final_model.joblib').read_bytes()).hexdigest() == metadata['model_sha256']
saved = joblib.load(artifact / 'final_model.joblib')
test = read_csv(root / 'office_test.csv')
_, _, X, _ = submission_identifiers(test, read_csv(root / 'legacy/submission.csv'))
probabilities = {name: aligned_probabilities(pipeline, X.iloc[:100])
                 for name, pipeline in saved['models'].items()}
p = combine(probabilities, saved['weights']) if saved['weights'] is not None else probabilities[saved['selected_model']]
with np.load(artifact / 'submission_probabilities.npz') as expected:
    np.testing.assert_allclose(p, expected['probabilities'][:100], atol=1e-12, rtol=0)
print('PASS: reloaded fitted model reproduces first 100 saved unlabeled-test probability rows.')
