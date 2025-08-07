#!/usr/bin/env python3
"""
Test script for the Washington State voter demographics downloader.

This script demonstrates how to use the voter demographics downloader
to download the voter demographics tables from the Washington State
Secretary of State website.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.downloader import WAStateDownloader
from utils.constants_WA import VOTER_DEMOGRAPHICS_URL, VOTER_DEMOGRAPHICS_FILENAME


def test_voter_demographics_download():
    """
    Test the voter demographics downloader functionality.
    """
    print("Washington State Voter Demographics Downloader Test")
    print("=" * 60)
    
    # Initialize the downloader
    downloader = WAStateDownloader(output_dir="data/raw/wa")
    
    # Display the URL and filename constants
    print(f"Voter Demographics URL: {VOTER_DEMOGRAPHICS_URL}")
    print(f"Local Filename: {VOTER_DEMOGRAPHICS_FILENAME}")
    print()
    
    # Download the voter demographics data
    print("Downloading voter demographics data...")
    demographics_path = downloader.download_voter_demographics()
    
    if demographics_path:
        print(f"✓ SUCCESS: Voter demographics downloaded successfully!")
        print(f"  File path: {demographics_path}")
        print(f"  File size: {demographics_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Try to read the Excel file to verify it's valid
        try:
            import pandas as pd
            df = pd.read_excel(demographics_path)
            print(f"  Excel file validation: {len(df)} rows, {len(df.columns)} columns")
            print(f"  Column names: {list(df.columns)}")
        except Exception as e:
            print(f"  Warning: Could not read Excel file: {e}")
        
        return True
    else:
        print("✗ FAILED: Could not download voter demographics data")
        return False


def main():
    """
    Main function to run the test.
    """
    success = test_voter_demographics_download()
    
    if success:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("The voter demographics data has been downloaded to:")
        print("  data/raw/wa/demographics/wa_voter_demographics_tables.xlsx")
    else:
        print("\n" + "=" * 60)
        print("Test failed!")
        print("Please check the error messages above for more information.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 