"""Random-Forest product recommender using the stored pairwise training dataset.

ROLE
----
The training CSV contains one row per (client mandate, candidate product) pair.
`selected` is the binary target: 1 if that candidate product is suitable for the
mandate, 0 otherwise.

PROJECT SCOPE
--------
The project is single-asset only. The training CSV keeps its pairwise schema
and the model trains only on rows compatible with that scope:
- n_underlyings == 1
- basket_preference == 0
- worst_of_requested == 0
- supports_single_asset == 1
- supports_multi_asset == 0

MODEL
-----
A fixed RandomForestClassifier trained from the stored labelled pairwise dataset.
No hyperparameter search is performed inside this module.

INFERENCE
---------
For one ClientMandate, we create one candidate row for every supported
single-asset product, score P(selected=1 | mandate, product), sort descending,
and return the Top 5. Recommendation Score = that Random-Forest probability.
"""

import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .schemas import ProductRecommendation


MODEL_PATH = "models/recommendation_model.pkl"
METRICS_PATH = "models/metrics.json"
TRAINING_PATH = "data/model_training/recommender_training.csv"

# Exact schema expected from the user's recommender_training.csv.
MANDATE_COLUMNS = [
    "client_type",
    "target_return",
    "maturity",
    "downside_tolerance",
    "income_preference",
    "autocall_acceptance",
    "basket_preference",
    "worst_of_requested",
    "memory_requested",
    "n_underlyings",
    "risk_appetite",
]

PRODUCT_COLUMNS = [
    "target_label",
    "family",
    "supports_income",
    "autocallable",
    "supports_single_asset",
    "supports_multi_asset",
    "supports_worst_of",
    "supports_memory",
]

FEATURE_COLUMNS = MANDATE_COLUMNS + PRODUCT_COLUMNS
TARGET_COLUMN = "selected"

CATEGORICAL_COLUMNS = [
    "client_type",
    "risk_appetite",
    "target_label",
    "family",
]

NUMERIC_COLUMNS = [
    "target_return",
    "maturity",
    "downside_tolerance",
    "income_preference",
    "autocall_acceptance",
    "basket_preference",
    "worst_of_requested",
    "memory_requested",
    "n_underlyings",
    "supports_income",
    "autocallable",
    "supports_single_asset",
    "supports_multi_asset",
    "supports_worst_of",
    "supports_memory",
]

# The construction/pricing engine uses these names.
TARGET_TO_ENGINE_FAMILY = {
    "Vanilla Option - Single Asset": "Vanilla",
    "Digital - Single Asset": "Digital",
    "Barrier - Single Asset": "Barrier",
    "Reverse Convertible - Single Asset": "Reverse Convertible",
    "ELN - Single Asset": "ELN",
    "Athena - Single Asset": "Athena",
    "Phoenix - Single Asset": "Phoenix",
    "Memory Phoenix - Single Asset": "Memory Phoenix",
    "Range Accrual - Single Asset": "Range Accrual",
    "Strip of Digitals - Single Asset": "Strip of Digitals",
}


def _validate_schema(df: pd.DataFrame) -> None:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "recommender_training.csv is missing required columns: "
            + ", ".join(missing)
        )


def _single_asset_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict the stored training CSV to the project single-asset universe."""
    mask = (
        (pd.to_numeric(df["n_underlyings"], errors="coerce") == 1)
        & (pd.to_numeric(df["basket_preference"], errors="coerce") == 0)
        & (pd.to_numeric(df["worst_of_requested"], errors="coerce") == 0)
        & (pd.to_numeric(df["supports_single_asset"], errors="coerce") == 1)
        & (pd.to_numeric(df["supports_multi_asset"], errors="coerce") == 0)
        & (df["target_label"].isin(TARGET_TO_ENGINE_FAMILY.keys()))
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError("No single-asset training rows remain after scope filtering.")
    return out


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the stored CSV / inference rows to the model feature schema."""
    out = df.copy()

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].fillna("").astype(str)

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out[FEATURE_COLUMNS]


def _build_pipeline() -> Pipeline:
    """Fixed preprocessing + fixed Random Forest. No hyperparameter search."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("features", preprocessor),
            ("model", model),
        ]
    )


def _mandate_group_key(df: pd.DataFrame) -> pd.Series:
    """Group identical mandate rows so the same mandate is not in train and test."""
    return df[MANDATE_COLUMNS].astype(str).agg("|".join, axis=1)


def _candidate_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """One static capability row for each supported single-asset product."""
    static_cols = PRODUCT_COLUMNS
    catalog = df[static_cols].drop_duplicates().copy()

    # One target_label must map to one unique static product definition.
    counts = catalog.groupby("target_label").size()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(
            "Inconsistent product capability rows for: "
            + ", ".join(bad.index.astype(str))
        )

    return catalog.sort_values("target_label").reset_index(drop=True)


def train_and_save(training_path: str = TRAINING_PATH):
    """Train the binary suitability model on the user's stored pairwise dataset."""
    raw = pd.read_csv(training_path)
    _validate_schema(raw)

    df = _single_asset_rows(raw)
    X = _clean_features(df)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)

    if set(y.unique()) - {0, 1}:
        raise ValueError("'selected' must be binary (0/1).")

    # Split by mandate, not by pair-row, so the same mandate does not
    # appear in both train and test through different candidate products.
    groups = _mandate_group_key(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    selected_index = list(pipe.classes_).index(1)
    proba_selected = pipe.predict_proba(X_test)[:, selected_index]

    metrics = {
        "model": "RandomForestClassifier",
        "scope": "single_asset_only",
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba_selected)),
        "training_rows_after_single_asset_filter": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "supported_products": int(df["target_label"].nunique()),
        "note": (
            "Uses the stored pairwise recommender_training.csv; "
            "no data generation and no GridSearch."
        ),
    }

    bundle = {
        "model": pipe,
        "candidate_catalog": _candidate_catalog(df),
        "feature_columns": FEATURE_COLUMNS,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return bundle, metrics


def load_model():
    """Load model + single-asset candidate catalog saved by train_and_save()."""
    return joblib.load(MODEL_PATH)


def _mandate_values(mandate) -> Dict:
    """Convert ClientMandate fields to the exact mandate features in the CSV."""
    return {
        "client_type": mandate.client_type,
        "target_return": 0.0 if mandate.target_return is None else mandate.target_return,
        "maturity": 0.0 if mandate.maturity_years is None else mandate.maturity_years,
        "downside_tolerance": (
            0.0 if mandate.downside_tolerance is None else mandate.downside_tolerance
        ),
        "income_preference": int(bool(mandate.income_preference)),
        "autocall_acceptance": int(bool(mandate.autocall_acceptance)),
        # Project scope is explicitly single-asset.
        "basket_preference": 0,
        "worst_of_requested": 0,
        "memory_requested": int(bool(mandate.memory_requested)),
        "n_underlyings": 1,
        "risk_appetite": mandate.risk_appetite or "medium",
    }


def _inference_frame(mandate, candidate_catalog: pd.DataFrame) -> pd.DataFrame:
    """Create one (mandate, candidate product) row per supported product."""
    common = _mandate_values(mandate)
    rows: List[Dict] = []

    for _, product in candidate_catalog.iterrows():
        row = dict(common)
        for col in PRODUCT_COLUMNS:
            row[col] = product[col]
        rows.append(row)

    return _clean_features(pd.DataFrame(rows))


def predict_top5(model_bundle, mandate, top_k: int = 5):
    """Score every supported single-asset product and return the best candidates."""
    pipe = model_bundle["model"]
    catalog = model_bundle["candidate_catalog"].reset_index(drop=True)

    X = _inference_frame(mandate, catalog)
    selected_index = list(pipe.classes_).index(1)
    scores = pipe.predict_proba(X)[:, selected_index]

    order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    recommendations = []

    for i in order:
        target_label = str(catalog.loc[i, "target_label"])
        engine_family = TARGET_TO_ENGINE_FAMILY[target_label]
        recommendations.append(
            ProductRecommendation(
                product_family=engine_family,
                score=float(scores[i]),
            )
        )

    return recommendations
