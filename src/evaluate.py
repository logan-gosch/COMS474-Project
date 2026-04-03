import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import OUTPUT_DIR, RANDOM_STATE

def run_regression(X, y, numeric_features, models, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        results.append({
            "model": model_name,
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        })

    return pd.DataFrame(results).sort_values(by="r2", ascending=False)

def run_classification(X, y, numeric_features, models, preprocessor):
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    reports = {}

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, preds),
        })

        reports[model_name] = classification_report(y_test, preds)

    return pd.DataFrame(results).sort_values(by="accuracy", ascending=False), reports

def save_results(reg_results, cls_results, cls_reports):
    OUTPUT_DIR.mkdir(exist_ok=True)

    reg_results.to_csv(OUTPUT_DIR / "regression_results.csv", index=False)
    cls_results.to_csv(OUTPUT_DIR / "classification_results.csv", index=False)

    with open(OUTPUT_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
        for model_name, report in cls_reports.items():
            f.write(f"{model_name}\n")
            f.write("=" * len(model_name) + "\n")
            f.write(report)
            f.write("\n\n")

    summary = {
        "best_regression_model": reg_results.iloc[0]["model"],
        "best_regression_r2": float(reg_results.iloc[0]["r2"]),
        "best_classification_model": cls_results.iloc[0]["model"],
        "best_classification_accuracy": float(cls_results.iloc[0]["accuracy"]),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)