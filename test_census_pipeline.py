import pandas as pd

from scr.censuspipeline.reduction import filter_variables, remove_high_correlation
from scr.censuspipeline.pipeline import ACSFeatureReductionPipeline
import scr.censuspipeline.metadata as md


def test_filter_variables_keywords():
    metadata = pd.DataFrame(
        [
            {"name": "B01001_001E", "label": "Total population", "concept": "SEX BY AGE"},
            {"name": "B25003_001E", "label": "Total housing units", "concept": "TENURE"},
        ]
    )
    filtered = filter_variables(metadata, keywords=["housing"])
    assert list(filtered["name"]) == ["B25003_001E"]


def test_remove_high_correlation():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],  # perfect correlation with a
            "c": [1, 1, 2, 2],
        }
    )
    reduced = remove_high_correlation(df, threshold=0.95)
    assert "b" not in reduced.columns
    assert "a" in reduced.columns


def test_pipeline_basic():
    # small metadata
    metadata = pd.DataFrame(
        [
            {"name": "AGE", "label": "Age", "concept": "Demographics"},
            {"name": "INCOME", "label": "Income", "concept": "Economics"},
        ]
    )
    pipeline = ACSFeatureReductionPipeline()
    pipeline._metadata = metadata
    pipeline.select_variables()
    df = pd.DataFrame(
        {
            "AGE": [10, 20, 30, 40, 50],
            "INCOME": [100, 180, 120, 160, 130],
        }
    )
    reduced = pipeline.reduce_dataframe(df, corr_threshold=0.9)
    assert set(reduced.columns) == {"AGE", "INCOME"}


def test_pipeline_common_variables(monkeypatch):
    def fake_fetch_variable_metadata(year, dataset):
        if year == 2012:
            return pd.DataFrame(
                [
                    {"name": "A", "label": "A", "concept": ""},
                    {"name": "B", "label": "B", "concept": ""},
                ]
            )
        return pd.DataFrame(
            [
                {"name": "B", "label": "B", "concept": ""},
                {"name": "C", "label": "C", "concept": ""},
            ]
        )

    monkeypatch.setattr(md, "fetch_variable_metadata", fake_fetch_variable_metadata)
    pipeline = ACSFeatureReductionPipeline(years=[2012, 2013])
    meta = pipeline.load_metadata()
    assert list(meta["name"]) == ["B"]
