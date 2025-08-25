import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from src.models.train_elasticnet import (
    prepare_data,
    train_elasticnet,
    hyperparameter_tuning,
    evaluate_model,
    save_model,
)


def test_elasticnet_workflow(tmp_path):
    # Create synthetic dataset
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42
    )

    # Train model
    model = train_elasticnet(X_train, y_train, alpha=0.1, l1_ratio=0.5)
    assert isinstance(model, ElasticNet)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    assert set(metrics.keys()) == {"mse", "r2"}

    # Hyperparameter tuning
    param_grid = {"alpha": [0.01, 0.1], "l1_ratio": [0.1, 0.5]}
    best_model = hyperparameter_tuning(X_train, y_train, param_grid)
    assert isinstance(best_model, ElasticNet)

    tuned_metrics = evaluate_model(best_model, X_test, y_test)
    assert set(tuned_metrics.keys()) == {"mse", "r2"}

    # Save model
    model_path = tmp_path / "elasticnet.joblib"
    save_model(best_model, str(model_path))
    assert model_path.exists()

