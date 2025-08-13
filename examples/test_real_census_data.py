#!/usr/bin/env python3
"""
Test script to demonstrate fetching real census data for Washington congressional districts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.downloader import CensusDownloader
import pandas as pd

def test_real_census_data():
    """Test fetching real census data for Washington congressional districts."""
    print("Testing Real Census Data Fetching")
    print("=" * 50)
    
    # Initialize the census downloader
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Test with a recent year (2020)
    print("Testing single year download (2020)...")
    df_2020 = census_downloader.download_wa_congressional_districts(2020)
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded REAL census data for 2020")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Columns: {list(df_2020.columns)}")
        print(f"  Congressional Districts: {list(df_2020['NAME'].unique())}")
        print(f"  Sample data:")
        print(df_2020.head())
        
        # Show some actual census values
        print(f"\nReal Census Data Sample:")
        for col in df_2020.columns:
            if col not in ['NAME', 'year']:
                print(f"  {col}: {df_2020[col].iloc[0]:,.0f}")
    else:
        print("✗ Failed to download real census data for 2020")
        print("  This might be due to:")
        print("  - Missing censusdata package (pip install censusdata)")
        print("  - Network connectivity issues")
        print("  - Census API rate limits")
    
    # Test downloading and combining multiple years
    print("\n" + "=" * 50)
    print("Testing combined download (2018-2020)...")
    
    years_to_test = [2018, 2019, 2020]
    combined_data = []
    
    for year in years_to_test:
        print(f"Downloading {year}...")
        df = census_downloader.download_wa_congressional_districts(year)
        if df is not None:
            combined_data.append(df)
            print(f"  ✓ {year}: {df.shape[0]} districts")
        else:
            print(f"  ✗ {year}: Failed")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"\n✓ Successfully combined {len(combined_data)} years of real census data")
        print(f"  Total shape: {combined_df.shape}")
        print(f"  Years: {sorted(combined_df['year'].unique())}")
        print(f"  Districts: {list(combined_df['NAME'].unique())}")
    else:
        print("\n✗ No real census data was successfully downloaded")

if __name__ == "__main__":
    test_real_census_data()
