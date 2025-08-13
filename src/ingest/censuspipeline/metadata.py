import pandas as pd
import requests
from typing import Any, Dict, Sequence


def fetch_variable_metadata(year: int = 2023, dataset: str = "acs/acs5") -> pd.DataFrame:
    """Fetch metadata for ACS variables from the Census API.

    Parameters
    ----------
    year: int, default 2023
        ACS year.
    dataset: str, default "acs/acs5"
        Dataset to query. See https://api.census.gov/data.html for options.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing variable name, label and concept.
    """
    url = f"https://api.census.gov/data/{year}/{dataset}/variables.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()["variables"]
    records = []
    for name, details in payload.items():
        record = {
            "name": name,
            "label": details.get("label", ""),
            "concept": details.get("concept", ""),
        }
        records.append(record)
    return pd.DataFrame(records)


def fetch_common_variable_metadata(
    years: Sequence[int], dataset: str = "acs/acs5"
) -> pd.DataFrame:
    """Fetch variables that are available across multiple ACS years.

    Parameters
    ----------
    years: Sequence[int]
        Years to intersect. Only variables present in all years are kept.
    dataset: str, default "acs/acs5"
        Dataset to query. See https://api.census.gov/data.html for options.

    Returns
    -------
    pandas.DataFrame
        Metadata for variables common to all provided years, using the
        metadata from the most recent year for labels and concepts.
    """
    if not years:
        raise ValueError("years must be a non-empty sequence")

    md_by_year = {year: fetch_variable_metadata(year, dataset) for year in years}
    common_names = set.intersection(*(set(df["name"]) for df in md_by_year.values()))
    latest_year = max(years)
    latest_md = md_by_year[latest_year]
    return latest_md[latest_md["name"].isin(common_names)].reset_index(drop=True)
