#!/usr/bin/env python3
"""
Example script demonstrating how to use the generic CensusDownloader
for multiple states and geographic levels.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.downloader import CensusDownloader
from utils.constants_WA import CENSUS_YEARS, CENSUS_VARIABLES
import censusdata
import pandas as pd

def download_california_counties():
    """
    Example: Download census data for California counties.
    """
    print("California Counties Example:")
    print("=" * 40)
    
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Define California counties geographic specification
    ca_counties_geo = censusdata.censusgeo([("state", "06"), ("county", "*")])
    
    # Download data for 2020
    df_2020 = census_downloader.download_census_data(
        year=2020,
        geo=ca_counties_geo,
        state_name="ca",
        variables=CENSUS_VARIABLES,
        geographic_level="county"
    )
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded California counties data for 2020")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Counties: {sorted(df_2020['NAME'].unique())}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download California counties data")
    
    return df_2020

def download_texas_congressional_districts():
    """
    Example: Download census data for Texas congressional districts.
    """
    print("\nTexas Congressional Districts Example:")
    print("=" * 40)
    
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Define Texas congressional districts geographic specification
    tx_cd_geo = censusdata.censusgeo([("state", "48"), ("congressional district", "*")])
    
    # Download data for 2020
    df_2020 = census_downloader.download_census_data(
        year=2020,
        geo=tx_cd_geo,
        state_name="tx",
        variables=CENSUS_VARIABLES,
        geographic_level="congressional district"
    )
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded Texas congressional districts data for 2020")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Congressional Districts: {sorted(df_2020['NAME'].unique())}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download Texas congressional districts data")
    
    return df_2020

def download_new_york_zip_codes():
    """
    Example: Download census data for New York ZIP code tabulation areas.
    """
    print("\nNew York ZIP Code Tabulation Areas Example:")
    print("=" * 40)
    
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Define New York ZIP code tabulation areas geographic specification
    ny_zip_geo = censusdata.censusgeo([("state", "36"), ("zip code tabulation area", "*")])
    
    # Download data for 2020
    df_2020 = census_downloader.download_census_data(
        year=2020,
        geo=ny_zip_geo,
        state_name="ny",
        variables=CENSUS_VARIABLES,
        geographic_level="zip code tabulation area"
    )
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded New York ZIP code data for 2020")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Number of ZIP codes: {len(df_2020['NAME'].unique())}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download New York ZIP code data")
    
    return df_2020

def download_multiple_variables():
    """
    Example: Download census data with multiple variables.
    """
    print("\nMultiple Variables Example:")
    print("=" * 40)
    
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Define multiple census variables
    variables = [
        "B01001_001E",  # Total population
        "B19013_001E",  # Median household income
        "B08303_001E",  # Commuting time
        "B25077_001E",  # Median home value
        "B15003_022E",  # Bachelor's degree
        "B15003_023E",  # Master's degree
        "B15003_024E",  # Professional degree
        "B15003_025E",  # Doctorate degree
    ]
    
    # Download Washington congressional districts with multiple variables
    df_2020 = census_downloader.download_wa_congressional_districts(2020, variables=variables)
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded Washington congressional districts data with multiple variables")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Variables: {list(df_2020.columns)}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download Washington congressional districts data")
    
    return df_2020

def main():
    """
    Main function to run all examples.
    """
    print("Multi-State Census Data Download Examples")
    print("=" * 60)
    
    try:
        # Run Washington example (using the convenience method)
        print("\nWashington State Example (using convenience method):")
        print("-" * 50)
        census_downloader = CensusDownloader(output_dir="data/raw/census")
        wa_df = census_downloader.download_wa_congressional_districts(2020)
        if wa_df is not None:
            print(f"✓ Washington congressional districts: {wa_df.shape}")
        
        # Run other state examples
        ca_df = download_california_counties()
        tx_df = download_texas_congressional_districts()
        ny_df = download_new_york_zip_codes()
        multi_var_df = download_multiple_variables()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        print(f"Washington congressional districts: {'✓' if wa_df is not None else '✗'}")
        print(f"California counties: {'✓' if ca_df is not None else '✗'}")
        print(f"Texas congressional districts: {'✓' if tx_df is not None else '✗'}")
        print(f"New York ZIP codes: {'✓' if ny_df is not None else '✗'}")
        print(f"Multiple variables: {'✓' if multi_var_df is not None else '✗'}")
        
        # Print download status
        print("\nDownload Status:")
        status = census_downloader.get_download_status()
        print(f"Census directory: {status['census_directory']}")
        print("Files downloaded:")
        for filename, info in status['files'].items():
            print(f"  {filename}: {info['size_mb']} MB")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages:")
        print("pip install censusdata pandas")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 