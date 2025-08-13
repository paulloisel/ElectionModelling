"""Utilities for selecting variables using the OpenAI API."""

from __future__ import annotations

import os
from typing import Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may not be installed
    OpenAI = None  # type: ignore


class OpenAISelector:
    """Select important variables via an OpenAI model.

    This class requires the ``openai`` package and an API key supplied via the
    ``OPENAI_API_KEY`` environment variable. The API call is intentionally kept
    simple so the class can be mocked easily in tests.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAISelector but is not installed"
            )
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAISelector")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def select_variables(
        self, variables: List[Dict[str, str]], top_k: int = 20
    ) -> List[str]:
        """Use an LLM to choose a subset of variable codes.

        Parameters
        ----------
        variables: list of dict
            Each dict must have ``name`` and ``label`` keys.
        top_k: int
            Number of variables to request from the model.

        Returns
        -------
        list[str]
            Selected variable codes as returned by the model.
        """
        prompt_lines = [
            "Select the top {} variables useful for predicting voter turnout.".format(
                top_k
            ),
            "Return a comma separated list of variable codes only.",
            "Available variables:",
        ]
        for var in variables:
            prompt_lines.append(f"{var['name']}: {var['label']}")
        prompt = "\n".join(prompt_lines)
        response = self.client.responses.create(model=self.model, input=prompt)
        # The API returns content in a nested structure; extract text
        text = response.output[0].content[0].text
        codes = [c.strip() for c in text.split(",")]
        valid = {v["name"] for v in variables}
        return [c for c in codes if c in valid]
