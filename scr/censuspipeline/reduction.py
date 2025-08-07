"""Utilities for reducing ACS variable sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence


def filter_variables(
    metadata: pd.DataFrame,
    keywords: Optional[Sequence[str]] = None,
    table_prefixes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter variable metadata by keyword or table prefix.

    Parameters
    ----------
    metadata: DataFrame
        Metadata as returned by :func:`fetch_variable_metadata`.
    keywords: sequence of str, optional
        Keep variables whose ``label`` or ``concept`` contains any keyword.
    table_prefixes: sequence of str, optional
        Keep variables whose name starts with one of these prefixes
        (e.g., ``["B01001", "B25003"]``).
    """
    result = metadata
    if keywords:
        mask = pd.Series(False, index=result.index)
        for kw in keywords:
            mask = mask | result["label"].str.contains(kw, case=False, na=False)
            mask = mask | result["concept"].str.contains(kw, case=False, na=False)
        result = result[mask]
    if table_prefixes:
        mask = pd.Series(False, index=result.index)
        for prefix in table_prefixes:
            mask = mask | result["name"].str.startswith(prefix)
        result = result[mask]
    return result.reset_index(drop=True)


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Remove highly correlated columns from ``df``.

    Parameters
    ----------
    df: DataFrame
        Data with numeric columns to check for correlation.
    threshold: float, default 0.9
        Drop one of two columns when correlation coefficient exceeds this value.

    Returns
    -------
    DataFrame
        ``df`` with correlated columns removed.
    """
    if df.empty:
        return df
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
