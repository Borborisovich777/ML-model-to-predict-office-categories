"""Run from the repository root: python verification/check_smoke.py."""
import json
from pathlib import Path
import numpy as np

roots = [Path('artifacts-smoke'), Path('artifacts-smoke-repeat')]
a, b = [json.loads((p / 'smoke_metrics.json').read_text()) for p in roots]
assert a['cv'] == b['cv']
assert a['selected_model'] == b['selected_model']
assert not a['holdout_evaluated'] and not b['holdout_evaluated']
with np.load(roots[0] / 'development_oof.npz') as x, np.load(roots[1] / 'development_oof.npz') as y:
    for key in x.files:
        np.testing.assert_array_equal(x[key], y[key])
print('PASS: identical smoke probabilities, folds, scores and selection; no holdout evaluation.')

with np.load('artifacts-smoke/development_oof.npz') as x, np.load('artifacts-smoke-parallel/development_oof.npz') as y:
    for key in x.files:
        np.testing.assert_array_equal(x[key], y[key])
parallel = json.loads(Path('artifacts-smoke-parallel/smoke_metrics.json').read_text())
assert a['cv'] == parallel['cv']
assert a['selected_model'] == parallel['selected_model']
assert not parallel['holdout_evaluated']
print('PASS: parallel smoke results also match exactly.')
