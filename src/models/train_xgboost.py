"""XGBoost Model Training

Trains XGBoost models for election turnout prediction.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def prepare_data(
    df: pd.DataFrame, target_col: str, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for XGBoost training.

    Parameters
    ----------
    df:
        Input ``pandas`` DataFrame containing both features and target.
    target_col:
        Name of the column containing the target variable.
    feature_cols:
        List of column names to be used as features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix ``X`` and target vector ``y`` as numpy arrays.
    """

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    return X, y


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any] | None = None
) -> xgb.XGBRegressor:
    """Train an ``xgboost.XGBRegressor`` model.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    params:
        Dictionary of parameters to pass to ``XGBRegressor``. If ``None`` a set of
        reasonable defaults is used.

    Returns
    -------
    xgb.XGBRegressor
        Trained regressor instance.
    """

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "random_state": 42,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[Any]],
) -> xgb.XGBRegressor:
    """Perform hyperparameter tuning using ``GridSearchCV``.

    Parameters
    ----------
    X_train, y_train:
        Training data.
    param_grid:
        Grid of parameters to search over.

    Returns
    -------
    xgb.XGBRegressor
        Best estimator found during the search.
    """

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(
    model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate a trained XGBoost model.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean squared error (``mse``) and ``r2`` score.
    """

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "r2": r2}


def feature_importance_analysis(
    model: xgb.XGBRegressor, feature_names: List[str]
) -> pd.DataFrame:
    """Generate a DataFrame with feature importance scores.

    Parameters
    ----------
    model:
        Trained XGBoost model.
    feature_names:
        Names of the features in the order used for training.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``feature`` and ``importance`` sorted by descending
        importance.
    """

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model: xgb.XGBRegressor, model_path: str) -> None:
    """Persist a trained model to disk using ``joblib``."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    # Example usage
    pass

