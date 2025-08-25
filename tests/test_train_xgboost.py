import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.models.train_xgboost import (
    evaluate_model,
    feature_importance_analysis,
    hyperparameter_tuning,
    prepare_data,
    save_model,
    train_xgboost,
)


def test_xgboost_workflow(tmp_path):
    """End-to-end test of the XGBoost training utilities."""

    # Synthetic dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    coef = np.array([1.5, -2.0])
    y = X @ coef + rng.normal(scale=0.1, size=100)
    df = pd.DataFrame(X, columns=["feat1", "feat2"])
    df["target"] = y

    # Prepare data
    X_arr, y_arr = prepare_data(df, "target", ["feat1", "feat2"])
    assert X_arr.shape == (100, 2)
    assert y_arr.shape == (100,)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )

    # Train model
    params = {"n_estimators": 10, "max_depth": 3, "objective": "reg:squarederror"}
    model = train_xgboost(X_train, y_train, params=params)
    assert isinstance(model, XGBRegressor)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics.keys()) == {"mse", "r2"}

    # Hyperparameter tuning
    param_grid = {
        "max_depth": [2, 3],
        "n_estimators": [5, 10],
        "objective": ["reg:squarederror"],
    }
    best_model = hyperparameter_tuning(X_train, y_train, param_grid)
    assert isinstance(best_model, XGBRegressor)

    tuned_metrics = evaluate_model(best_model, X_test, y_test)
    assert set(tuned_metrics.keys()) == {"mse", "r2"}

    # Feature importance
    importances = feature_importance_analysis(best_model, ["feat1", "feat2"])
    assert set(importances.columns) == {"feature", "importance"}
    assert len(importances) == 2

    # Save model
    model_path = tmp_path / "xgb.joblib"
    save_model(best_model, str(model_path))
    assert model_path.exists()

