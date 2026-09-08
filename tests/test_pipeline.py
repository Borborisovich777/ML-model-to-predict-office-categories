import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from office_ml.config import CANDIDATES, CLASSES, SINGLES, WEIGHTS, model_parameters
from office_ml.evaluation import (aligned_probabilities, combine, folds, select_candidate,
                                  split_indices, validate_probabilities)
from office_ml.preprocessing import FoldPreprocessor, engineer_features, make_pipeline
from office_ml.train import read_csv, submission_identifiers


@pytest.fixture
def training():
    return pd.DataFrame({"number": [1., 3., 5., np.nan],
                         "all_missing": [np.nan] * 4,
                         "category": ["a", "b", "a", None]})


@pytest.mark.parametrize("native", [False, True])
def test_train_only_medians_and_validation_category_cannot_change_training(training, native):
    prep = FoldPreprocessor(native).fit(training, [0, 1, 2, 3])
    before = prep.transform(training)
    medians = prep.medians_.copy()
    validation = pd.DataFrame({"number": [1e12, np.nan], "all_missing": [10., np.nan],
                               "category": ["validation-only", None]})
    transformed = prep.transform(validation)
    assert_series_equal(prep.medians_, medians)
    assert medians['number'] == 3.0
    assert medians['all_missing'] == 0.0
    after = prep.transform(training)
    if native:
        assert_frame_equal(before, after)
        assert transformed.iloc[1]['number'] == 3
        assert transformed.iloc[0]['category'] == 'value:validation-only'
    else:
        np.testing.assert_array_equal(before, after)
        assert transformed.iloc[1, 0] == 3
        vocabulary = prep.encoder_.named_transformers_['categorical'].categories_[0]
        assert 'value:validation-only' not in vocabulary
        assert transformed.iloc[0, 2:].sum() == 0
    # Label permutation has no effect on externally transformed training features.
    relabeled = FoldPreprocessor(native).fit_transform(training, [4, 4, 4, 4])
    if native:
        assert_frame_equal(before, relabeled)
    else:
        np.testing.assert_array_equal(before, relabeled)


@pytest.mark.parametrize('native', [False, True])
def test_model_inputs_finite_for_invalid_ratios_and_missing_values(native):
    X = pd.DataFrame({'OfficeSpace': [1e308, -np.inf, np.nan, 0],
                      'PlotSize': [0, -1, np.nan, np.inf], 'Restrooms': [0, 0, 0, 0],
                      'MeetingRooms': [0, 0, 0, 0], 'BasementArea': [0, 1, 0, 0],
                      'ParkingArea': [0, 1, 0, 0], 'BuildingGrade': [1e308, 1, 1, 1],
                      'category': [None, 'a', 'b', '__MISSING__']})
    result = FoldPreprocessor(native).fit_transform(X)
    numeric = result.select_dtypes(include=np.number).to_numpy()
    assert np.isfinite(numeric).all()


def test_target_and_schema_guard(training):
    with pytest.raises(ValueError, match='Target'):
        FoldPreprocessor().fit(training.assign(OfficeCategory=0))
    fitted = FoldPreprocessor().fit(training)
    with pytest.raises(ValueError, match='schema'):
        fitted.transform(training.drop(columns='number'))


def test_probability_alignment_reorders_model_classes():
    class ReorderedModel:
        classes_ = np.array([4, 0, 3, 1, 2])
        def predict_proba(self, X):
            return np.tile([.5, .1, .2, .05, .15], (len(X), 1))
    p = aligned_probabilities(ReorderedModel(), [1, 2])
    np.testing.assert_allclose(p[0], [.1, .05, .15, .2, .5])


@pytest.mark.parametrize('name', SINGLES)
def test_actual_boosters_are_deterministic_and_probabilities_aligned(name):
    rng = np.random.default_rng(42)
    y = pd.Series(np.tile(CLASSES, 20))
    X = pd.DataFrame({'number': y + rng.normal(0, .3, len(y)),
                      'category': ['c' + str(v) for v in y]})
    params = model_parameters(smoke=True)
    a = make_pipeline(name, X, params).fit(X, y)
    b = make_pipeline(name, X, params).fit(X, y)
    validation = X.iloc[:10].copy()
    validation.loc[0, 'category'] = 'never-seen'
    pa, pb = aligned_probabilities(a, validation), aligned_probabilities(b, validation)
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_allclose(pa.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(a.classes_, CLASSES)
    assert pa.shape == (10, 5)


def test_weighted_soft_vote_and_probability_guards():
    p = {'catboost': np.array([[1, 0, 0, 0, 0]]),
         'xgboost': np.array([[0, 1, 0, 0, 0]]),
         'lightgbm': np.array([[0, 0, 1, 0, 0]])}
    np.testing.assert_allclose(combine(p, WEIGHTS['ensemble_weighted']), [[.375, .3125, .3125, 0, 0]])
    for invalid in (np.zeros((1, 5)), np.full((1, 5), np.nan), np.array([[2, -1, 0, 0, 0]])):
        with pytest.raises(ValueError):
            validate_probabilities(invalid)
    with pytest.raises(ValueError):
        combine(p, [1, -1, 1])


def test_split_folds_disjoint_exhaustive_stratified_and_deterministic():
    y = pd.Series(np.tile(CLASSES, 100))
    dev, holdout = split_indices(y)
    dev2, holdout2 = split_indices(y)
    np.testing.assert_array_equal(dev, dev2)
    np.testing.assert_array_equal(holdout, holdout2)
    assert not set(dev) & set(holdout)
    assert len(set(dev) | set(holdout)) == len(y)
    seen = []
    for (train, valid), (train2, valid2) in zip(folds(y.iloc[dev]), folds(y.iloc[dev])):
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(valid, valid2)
        assert not set(dev[train]) & set(holdout)
        assert not set(train) & set(valid)
        assert set(y.iloc[dev[valid]]) == set(CLASSES)
        seen.extend(valid)
    assert sorted(seen) == list(range(len(dev)))


def test_independent_iteration_budgets():
    params = model_parameters()
    assert params['catboost']['iterations'] == 300
    assert params['xgboost']['n_estimators'] == 240
    assert params['lightgbm']['n_estimators'] == 320
    assert all('early_stopping_rounds' not in p for p in params.values())


def test_selection_uses_development_scores_and_fixed_tie_order():
    s = {name: dict(accuracy=.2, macro_f1=.2) for name in CANDIDATES}
    assert select_candidate(s) == 'catboost'
    s['lightgbm']['accuracy'] = .7
    assert select_candidate(s) == 'lightgbm'


def test_preserve_source_and_template_identifiers():
    source = pd.DataFrame({'Id': [810, 4, 103], 'number': [1, 2, 3]})
    name, ids, features, _ = submission_identifiers(source, None)
    assert name == 'Id'
    np.testing.assert_array_equal(ids, [810, 4, 103])
    assert list(features) == ['number']
    template = pd.DataFrame({'Id': ['b-12', 'a-2', 'a-1'], 'OfficeCategory': [0, 0, 0]})
    _, ids, _, _ = submission_identifiers(features, template)
    np.testing.assert_array_equal(ids, template.Id)
    with pytest.raises(ValueError):
        submission_identifiers(features, None)
    with pytest.raises(ValueError):
        submission_identifiers(features, template.iloc[:2])
    with pytest.raises(ValueError):
        submission_identifiers(source.assign(Id=[4, 4, 4]), None)


def test_real_repository_submission_template_row_contract():
    root = Path(__file__).resolve().parents[1]
    test = pd.read_csv(root / 'office_test.csv')
    template = pd.read_csv(root / 'legacy/submission.csv')
    _, ids, X, provenance = submission_identifiers(test, template)
    np.testing.assert_array_equal(ids, template.Id)
    assert len(ids) == len(X) == 15000
    assert 'unverified' in provenance


def test_csv_preserves_leading_zero_identifiers(tmp_path):
    path = tmp_path / 'test.csv'
    path.write_text('Id,number\n0007,1\n0002,2\n')
    _, ids, _, _ = submission_identifiers(read_csv(path), None)
    np.testing.assert_array_equal(ids, ['0007', '0002'])


def test_evaluation_freezes_before_holdout_and_never_fits_holdout(tmp_path, monkeypatch):
    import office_ml.train as train
    import office_ml.reporting as reporting
    frame = pd.DataFrame({'number': np.arange(100), 'OfficeCategory': np.tile(CLASSES, 20)})
    frame.to_csv(tmp_path / 'office_train.csv', index=False)
    dev, holdout = split_indices(frame.OfficeCategory)
    holdout_set = set(holdout)
    output = tmp_path / 'artifacts'
    events = []

    class SpyPipeline:
        classes_ = np.asarray(CLASSES)
        def fit(self, X, y):
            assert not set(X.index) & holdout_set
            events.append(('fit', set(X.index)))
            return self
        def predict_proba(self, X):
            if set(X.index) & holdout_set:
                assert set(X.index) == holdout_set
                assert (output / 'frozen_selection.json').exists()
                frozen = json.loads((output / 'frozen_selection.json').read_text())
                assert frozen['selected_model'] == 'catboost'
                assert sum(e[0] == 'fit' and e[1] == set(dev) for e in events) == 4
                events.append(('holdout', set(X.index)))
            return np.full((len(X), 5), .2)

    monkeypatch.setattr(train, 'make_pipeline', lambda *a: SpyPipeline())
    monkeypatch.setattr(train, 'source_hashes', lambda *a: {})
    monkeypatch.setattr(train, 'versions', lambda: {})
    monkeypatch.setattr(reporting, 'plot_confusion_matrices', lambda *a: None)
    train.evaluate(tmp_path, output)
    assert sum(e[0] == 'holdout' for e in events) == 4
    before = (output / 'metrics.json').read_bytes()
    with pytest.raises(FileExistsError):
        train.evaluate(tmp_path, output)
    assert before == (output / 'metrics.json').read_bytes()


def test_parallel_refits_return_usable_fitted_pipelines():
    from office_ml.train import fit_models
    y = pd.Series(np.tile(CLASSES, 12))
    X = pd.DataFrame({'number': y.astype(float), 'category': y.astype(str)})
    fitted = fit_models(SINGLES, X, y, model_parameters(smoke=True), workers=2)
    assert [name for name, _, _ in fitted] == list(SINGLES)
    for name, pipeline, elapsed in fitted:
        probability = aligned_probabilities(pipeline, X.iloc[:10])
        assert probability.shape == (10, 5)
        assert elapsed > 0
