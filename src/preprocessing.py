import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS

def build_features(df: pd.DataFrame):
    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]

    X = df[feature_cols].copy()
    y_reg = np.log1p(df["Streams"])
    y_cls = pd.qcut(
        df["Streams"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    return X, y_reg, y_cls, feature_cols

def make_preprocessor(numeric_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )