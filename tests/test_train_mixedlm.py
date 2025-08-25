import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# Skip tests if statsmodels is not installed
pytest.importorskip("statsmodels")

from src.models.train_mixedlm import (
    prepare_mixedlm_data,
    train_mixedlm,
    evaluate_mixedlm,
    extract_random_effects,
    predict_mixedlm,
    save_model,
)


def test_mixedlm_workflow(tmp_path):
    rng = np.random.default_rng(0)

    n_groups = 5
    group_size = 20
    groups = np.repeat([f"g{i}" for i in range(n_groups)], group_size)
    n = n_groups * group_size

    feat1 = rng.normal(size=n)
    feat2 = rng.normal(size=n)

    # Create group-specific intercepts
    group_effects = rng.normal(scale=1.0, size=n_groups)
    intercept = np.array([group_effects[int(g[1:])] for g in groups])

    y = 1.0 + 2.0 * feat1 - 1.0 * feat2 + intercept + rng.normal(scale=0.1, size=n)

    df = pd.DataFrame({
        "target": y,
        "feat1": feat1,
        "feat2": feat2,
        "state": groups,
    })

    prepared_df, formula = prepare_mixedlm_data(
        df, "target", ["feat1", "feat2"], group_col="state"
    )
    assert formula == "target ~ feat1 + feat2"
    assert set(prepared_df.columns) == {"target", "feat1", "feat2", "state"}

    train_df, test_df = train_test_split(prepared_df, test_size=0.2, random_state=42)

    model = train_mixedlm(train_df, formula, groups="state")
    assert hasattr(model, "random_effects")

    metrics = evaluate_mixedlm(model, test_df, "target")
    assert set(metrics.keys()) == {"mse", "r2"}

    re_df = extract_random_effects(model)
    assert "group" in re_df.columns
    assert len(re_df) == n_groups

    preds = predict_mixedlm(model, test_df)
    assert preds.shape[0] == len(test_df)

    model_path = tmp_path / "mixedlm.joblib"
    save_model(model, str(model_path))
    assert model_path.exists()

