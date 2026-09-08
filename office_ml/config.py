"""Prospective settings: fixed before any repaired holdout performance is read."""

SEED = 42
HOLDOUT_SIZE = 0.20
FOLDS = 5
TARGET = "OfficeCategory"
CLASSES = (0, 1, 2, 3, 4)
SINGLES = ("catboost", "xgboost", "lightgbm")
WEIGHTS = {"ensemble_equal": (1.0, 1.0, 1.0),
           "ensemble_weighted": (1.2, 1.0, 1.0)}
# Order resolves exact ties, after accuracy and macro F1. No weight search.
CANDIDATES = (*SINGLES, *WEIGHTS, "dummy")
SELECTION_RULE = "highest development OOF accuracy, then macro F1, then candidate order"


def model_parameters(smoke=False):
    """Independent fixed budgets; no early stopping or best-iteration copying."""
    return {
        "catboost": dict(iterations=8 if smoke else 300, depth=6, learning_rate=0.08,
                         loss_function="MultiClass", l2_leaf_reg=3, border_count=128,
                         random_seed=SEED, thread_count=1, task_type="CPU",
                         verbose=False, allow_writing_files=False),
        "xgboost": dict(n_estimators=8 if smoke else 240, max_depth=5,
                        learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                        random_state=SEED, n_jobs=1, tree_method="hist"),
        "lightgbm": dict(n_estimators=8 if smoke else 320, max_depth=-1, num_leaves=31,
                         learning_rate=0.04, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, objective="multiclass", num_class=5,
                         random_state=SEED, n_jobs=1, deterministic=True,
                         force_col_wise=True, verbosity=-1),
        "dummy": dict(strategy="stratified", random_state=SEED),
    }
