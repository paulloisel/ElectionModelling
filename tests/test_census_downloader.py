#!/usr/bin/env python3
"""
Test script for the WACensusDownloader class.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.downloader import CensusDownloader
import pandas as pd

def test_census_downloader():
    """
    Test the census downloader functionality.
    """
    print("Testing CensusDownloader...")
    print("=" * 50)
    
    # Initialize the census downloader
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Test downloading data for a single year (2020)
    print("Testing single year download (2020)...")
    df_2020 = census_downloader.download_wa_congressional_districts(2020)
    
    if df_2020 is not None:
        print(f"✓ Successfully downloaded 2020 data")
        print(f"  Shape: {df_2020.shape}")
        print(f"  Columns: {list(df_2020.columns)}")
        print(f"  Number of Congressional Districts: {len(df_2020['NAME'].unique())}")
        print(f"  Sample data:")
        print(df_2020.head())
    else:
        print("✗ Failed to download 2020 data")
    
    # Test downloading and combining all years
    print("\nTesting combined download (2010-2023)...")
    combined_df = census_downloader.download_and_combine_wa_congressional_districts()
    
    if combined_df is not None:
        print(f"✓ Successfully downloaded and combined data")
        print(f"  Shape: {combined_df.shape}")
        print(f"  Years: {sorted(combined_df['year'].unique())}")
        print(f"  Number of Congressional Districts: {len(combined_df['NAME'].unique())}")
        print(f"  Sample data:")
        print(combined_df.head())
    else:
        print("✗ Failed to download and combine data")
    
    # Print download status
    print("\nDownload Status:")
    status = census_downloader.get_download_status()
    print(f"Census directory: {status['census_directory']}")
    print("Files downloaded:")
    for filename, info in status['files'].items():
        print(f"  {filename}: {info['size_mb']} MB")

if __name__ == "__main__":
    test_census_downloader() 