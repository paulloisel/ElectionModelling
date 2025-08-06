#!/usr/bin/env python3
"""
Example script demonstrating census data download for Washington State.
This script shows how to use the WACensusDownloader class and also includes
the original code provided by the user.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.downloader import CensusDownloader
from utils.constants_WA import CENSUS_YEARS, CENSUS_VARIABLES, CENSUS_BASE_URL
import censusdata
import pandas as pd

def original_census_code():
    """
    Original census download code as provided by the user.
    """
    print("Original Census Code Example:")
    print("=" * 50)
    
    # Constants (already defined in constants_WA.py)
    YEARS = CENSUS_YEARS      # range(2010, 2024)  # 2010-2023 5-year releases
    VARIABLES = CENSUS_VARIABLES  # ["B01001_001E"]    # add as many vars as you like
    BASE_URL = CENSUS_BASE_URL    # "https://api.census.gov/data/{yr}/acs/acs5"
    
    print(f"Years: {list(YEARS)}")
    print(f"Variables: {VARIABLES}")
    print(f"Base URL: {BASE_URL}")
    
    def download_wa(year, vars_=VARIABLES):
        geo = censusdata.censusgeo([("state", "53"), ("congressional district", "*")])
        df = censusdata.download("acs5", year, geo, vars_)
        df.reset_index(inplace=True)          # index → columns
        df.rename(columns={"index": "NAME"}, inplace=True)
        df["year"] = year
        return df
    
    # Download data for all years
    print("\nDownloading data for all years...")
    wa_cd = pd.concat([download_wa(y) for y in YEARS], ignore_index=True)
    
    print(f"Combined data shape: {wa_cd.shape}")
    print(f"Years included: {sorted(wa_cd['year'].unique())}")
    print(f"Congressional Districts: {sorted(wa_cd['NAME'].unique())}")
    print("\nSample data:")
    print(wa_cd.head())
    
    return wa_cd

def class_based_census_code():
    """
    Using the CensusDownloader class.
    """
    print("\nClass-Based Census Code Example:")
    print("=" * 50)
    
    # Initialize the census downloader
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Download data for a single year
    print("Downloading data for 2020...")
    df_2020 = census_downloader.download_wa_congressional_districts(2020)
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded 2020 data")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download 2020 data")
    
    # Download and combine all years
    print("\nDownloading and combining data for all years...")
    combined_df = census_downloader.download_and_combine_wa_congressional_districts()
    
    if combined_df is not None:
        print(f"✓ Successfully downloaded and combined data")
        print(f"  Shape: {combined_df.shape}")
        print(f"  Years: {sorted(combined_df['year'].unique())}")
        print(f"  Sample data:")
        print(combined_df.head())
    else:
        print("✗ Failed to download and combine data")
    
    return combined_df

def main():
    """
    Main function to run both examples.
    """
    print("Washington State Census Data Download Examples")
    print("=" * 60)
    
    try:
        # Run the original code example
        original_data = original_census_code()
        
        # Run the class-based example
        class_data = class_based_census_code()
        
        # Compare results
        if original_data is not None and class_data is not None:
            print("\nComparison:")
            print("=" * 30)
            print(f"Original code shape: {original_data.shape}")
            print(f"Class-based shape: {class_data.shape}")
            print(f"Dataframes are equal: {original_data.equals(class_data)}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages:")
        print("pip install censusdata pandas")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 