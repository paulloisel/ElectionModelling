"""Mixed Linear Model Training

Utility functions for fitting and evaluating mixed linear models with
group-specific intercepts using ``statsmodels``.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.metrics import mean_squared_error, r2_score


def prepare_mixedlm_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str = "state",
) -> Tuple[pd.DataFrame, str]:
    """Prepare data for mixed linear model training.

    Parameters
    ----------
    df:
        Input DataFrame containing target, features and grouping column.
    target_col:
        Name of the column containing the target variable.
    feature_cols:
        List of feature column names.
    group_col:
        Column used to denote grouping structure (e.g. state or county).

    Returns
    -------
    Tuple[pd.DataFrame, str]
        The subset DataFrame containing only the relevant columns and the
        Patsy-style formula string to be used with ``MixedLM``.
    """

    cols = [target_col] + feature_cols + [group_col]
    modeling_df = df[cols].dropna().copy()
    formula = f"{target_col} ~ {' + '.join(feature_cols)}"
    return modeling_df, formula


def train_mixedlm(df: pd.DataFrame, formula: str, groups: str) -> MixedLM:
    """Train a mixed linear model with group-specific intercepts.

    Parameters
    ----------
    df:
        Training DataFrame returned from :func:`prepare_mixedlm_data`.
    formula:
        Model formula expressed in Patsy syntax.
    groups:
        Column name in ``df`` that identifies group membership.

    Returns
    -------
    statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted model results object.
    """

    model = MixedLM.from_formula(formula, groups=groups, data=df)
    result = model.fit()
    return result


def evaluate_mixedlm(model: MixedLM, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """Evaluate a trained mixed linear model on a test dataset.

    Parameters
    ----------
    model:
        Fitted ``MixedLM`` results object.
    df:
        Test DataFrame containing the same columns used for training.
    target_col:
        Name of the target column.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean squared error (``mse``) and coefficient of
        determination (``r2``).
    """

    y_true = df[target_col].to_numpy()
    y_pred = model.predict(df)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "r2": r2}


def extract_random_effects(model: MixedLM) -> pd.DataFrame:
    """Extract random effects by group from a fitted model.

    Parameters
    ----------
    model:
        Fitted ``MixedLM`` results object.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the random effects with one row per group. The
        column names correspond to the random effect coefficients.
    """

    re_dict = model.random_effects
    re_df = pd.DataFrame.from_dict(re_dict, orient="index")
    re_df.index.name = "group"
    return re_df.reset_index()


def predict_mixedlm(model: MixedLM, df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a fitted mixed linear model."""

    return model.predict(df)


def save_model(model: MixedLM, model_path: str) -> None:
    """Persist a fitted model to disk using ``joblib``."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    # Example usage
    pass

