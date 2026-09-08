"""Target-free features and explicitly training-fitted preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


def engineer_features(frame):
    """Ten fixed row-local features; neither labels nor population statistics used.

    Undefined ratios and overflows become missing, then get training medians.
    Raw columns are retained. Ages use YearListed, never the wall clock.
    """
    out = frame.copy()
    numeric = out.select_dtypes(include=np.number).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)

    def ratio(name, numerator, denominator):
        if {numerator, denominator} <= set(out):
            out[name] = out[numerator] / out[denominator].where(out[denominator] > 0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if {"BuildingGrade", "OfficeSpace"} <= set(out):
            out["Quality_Size"] = out.BuildingGrade * out.OfficeSpace
        ratio("Space_Plot_Ratio", "OfficeSpace", "PlotSize")
        ratio("Space_Per_Restroom", "OfficeSpace", "Restrooms")
        ratio("Space_Per_MeetingRoom", "OfficeSpace", "MeetingRooms")
        if {"OfficeSpace", "BasementArea", "ParkingArea"} <= set(out):
            out["TotalArea"] = out.OfficeSpace + out.BasementArea + out.ParkingArea
            ratio("OfficeSpace_Pct", "OfficeSpace", "TotalArea")
            ratio("BasementArea_Pct", "BasementArea", "TotalArea")
        if {"YearListed", "ConstructionYear"} <= set(out):
            out["Age_At_Listing"] = out.YearListed - out.ConstructionYear
        if {"YearListed", "RenovationYear"} <= set(out):
            out["Years_Since_Renovation"] = out.YearListed - out.RenovationYear
            known = out.YearListed.notna() & out.RenovationYear.notna()
            out["Recent_Renovation"] = out.Years_Since_Renovation.between(0, 5).astype(float).where(known)
    numeric = out.select_dtypes(include=np.number).columns
    # Explicit bound prevents float32 overflow in downstream tree libraries.
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).clip(-1e15, 1e15)
    return out


class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Owns training schema, medians and, for one-hot mode, category vocabulary.

    Native CatBoost gets strings, including unseen strings, in categorical columns.
    One-hot models ignore unseen categories (all-zero indicator block).
    Labels are ignored: there is no external target encoding.
    """

    def __init__(self, native_categorical=False):
        self.native_categorical = native_categorical

    def fit(self, X, y=None):
        if "OfficeCategory" in X:
            raise ValueError("Target column must not be supplied as a predictor")
        self.input_columns_ = list(X.columns)
        frame = engineer_features(X)
        self.numeric_columns_ = list(frame.select_dtypes(include=np.number).columns)
        self.categorical_columns_ = [c for c in frame if c not in self.numeric_columns_]
        self.feature_columns_ = list(frame.columns)
        self.medians_ = frame[self.numeric_columns_].median().fillna(0.0)
        if not self.native_categorical:
            self.encoder_ = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
                 self.numeric_columns_),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                              dtype=np.float64), self.categorical_columns_),
            ], sparse_threshold=0, verbose_feature_names_out=False)
            self.encoder_.fit(self._strings(frame))
        return self

    def _strings(self, frame):
        frame = frame.copy()
        for column in self.categorical_columns_:
            # Prefix real values so the missing token cannot collide with a category.
            frame[column] = frame[column].map(
                lambda value: "__MISSING__" if pd.isna(value) else "value:" + str(value))
        return frame

    def transform(self, X):
        check_is_fitted(self, "medians_")
        if set(X.columns) != set(self.input_columns_):
            raise ValueError("Prediction columns differ from fitted training schema")
        frame = self._strings(engineer_features(X[self.input_columns_]))
        if self.native_categorical:
            frame[self.numeric_columns_] = frame[self.numeric_columns_].fillna(self.medians_)
            result = frame[self.feature_columns_]
            finite = np.isfinite(result[self.numeric_columns_].to_numpy()).all()
        else:
            result = self.encoder_.transform(frame)
            finite = np.isfinite(result).all()
            # Stable generic names avoid LightGBM's unnamed-array warning and
            # forbidden characters from user-supplied category strings.
            result = pd.DataFrame(result, index=X.index,
                                  columns=[f"f{i}" for i in range(result.shape[1])])
        if not finite:
            raise ValueError("Nonfinite model inputs after preprocessing")
        return result


def make_pipeline(name, X_fit, parameters):
    """Each returned pipeline starts unfitted; categorical names use training only."""
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.dummy import DummyClassifier

    native = name == "catboost"
    if native:
        engineered = engineer_features(X_fit)
        cats = [c for c in engineered if not pd.api.types.is_numeric_dtype(engineered[c])]
        model = CatBoostClassifier(**parameters[name], cat_features=cats)
    else:
        constructors = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier,
                        "dummy": DummyClassifier}
        model = constructors[name](**parameters[name])
    return Pipeline([("preprocess", FoldPreprocessor(native_categorical=native)),
                     ("model", model)])
