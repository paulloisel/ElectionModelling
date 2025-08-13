"""End-to-end feature reduction pipeline for ACS data."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .metadata import fetch_variable_metadata, fetch_common_variable_metadata
from .openai_selector import OpenAISelector
from .reduction import filter_variables, remove_high_correlation


class ACSFeatureReductionPipeline:
    """Pipeline that reduces the number of ACS variables.

    The pipeline performs the following steps:

    1. Download metadata for ACS variables, optionally keeping only variables
       present across multiple years.
    2. Filter variables by keyword or table prefix.
    3. Optionally use an OpenAI model to choose a final subset of variables.
    4. Remove highly correlated variables from a DataFrame of observations.
    """

    def __init__(
        self,
        year: int = 2023,
        years: Optional[Sequence[int]] = None,
        dataset: str = "acs/acs5",
        openai_selector: Optional[OpenAISelector] = None,
    ) -> None:
        self.year = year
        self.years = years
        self.dataset = dataset
        self.openai_selector = openai_selector
        self._metadata: Optional[pd.DataFrame] = None
        self._selected: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def load_metadata(self) -> pd.DataFrame:
        if self.years:
            self._metadata = fetch_common_variable_metadata(self.years, self.dataset)
        else:
            self._metadata = fetch_variable_metadata(self.year, self.dataset)
        return self._metadata

    # ------------------------------------------------------------------
    def select_variables(
        self,
        keywords: Optional[Sequence[str]] = None,
        table_prefixes: Optional[Sequence[str]] = None,
        openai_top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Select variables based on filters and optional OpenAI ranking."""
        if self._metadata is None:
            self.load_metadata()
        filtered = filter_variables(self._metadata, keywords, table_prefixes)
        if self.openai_selector and openai_top_k:
            vars_list = filtered[["name", "label"]].to_dict("records")
            chosen = self.openai_selector.select_variables(vars_list, openai_top_k)
            filtered = filtered[filtered["name"].isin(chosen)]
        self._selected = filtered
        return filtered

    # ------------------------------------------------------------------
    def reduce_dataframe(self, df: pd.DataFrame, corr_threshold: float = 0.9) -> pd.DataFrame:
        """Reduce a DataFrame to the selected variables and drop correlations."""
        if self._selected is None:
            raise ValueError("No variables selected. Call select_variables first.")
        vars_ = [v for v in self._selected["name"] if v in df.columns]
        reduced = remove_high_correlation(df[vars_], threshold=corr_threshold)
        return reduced
