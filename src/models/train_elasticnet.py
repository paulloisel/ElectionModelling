"""ElasticNet Model Training

Trains ElasticNet regression models for election turnout prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, List


def prepare_data(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for ElasticNet training.

    Args:
        df: Input DataFrame
        target_col: Target variable column
        feature_cols: Feature columns

    Returns:
        X, y arrays for training
    """
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    return X, y


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5
) -> ElasticNet:
    """Train ElasticNet model.

    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength
        l1_ratio: L1 ratio (0 = Ridge, 1 = Lasso)

    Returns:
        Trained ElasticNet model
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    return model


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, List[float]]
) -> ElasticNet:
    """Perform hyperparameter tuning using GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training targets
        param_grid: Parameter grid for tuning

    Returns:
        Best ElasticNet model
    """
    grid_search = GridSearchCV(
        ElasticNet(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(model: ElasticNet, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate ElasticNet model performance.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "r2": r2}


def save_model(model: ElasticNet, model_path: str) -> None:
    """Save trained model to disk.

    Args:
        model: Trained model
        model_path: Path to save model
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    # Example usage
    pass
