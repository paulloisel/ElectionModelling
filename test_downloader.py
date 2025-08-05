#!/usr/bin/env python3
"""
Test script for the WA State Downloader.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.downloader import WAStateDownloader

def test_downloader():
    """Test the WA State Downloader functionality."""
    
    print("Testing WA State Downloader...")
    
    # Initialize downloader with data/raw directory
    downloader = WAStateDownloader(output_dir="data/raw")
    
    # Test downloading data for a specific year (2020)
    print("\n1. Testing 2020 election data download...")
    year_2020 = downloader.download_specific_year_data(2020)
    
    print("Results for 2020:")
    for key, path in year_2020.items():
        status = "✓ SUCCESS" if path else "✗ FAILED"
        print(f"  {key}: {status}")
        if path:
            print(f"    File: {path}")
    
    # Test downloading data for all even years (just a few for testing)
    print("\n2. Testing download for years 2020 and 2024...")
    test_years = [2020, 2024]
    for year in test_years:
        print(f"\nProcessing year {year}...")
        year_data = downloader.download_specific_year_data(year)
        
        successful = sum(1 for path in year_data.values() if path is not None)
        total = len(year_data)
        print(f"  Year {year}: {successful}/{total} successful downloads")
    
    # Print overall status
    print("\n3. Overall download status:")
    status = downloader.get_download_status()
    print(f"Output directory: {status['output_directory']}")
    print("Files downloaded:")
    for filename, info in status['files'].items():
        print(f"  {filename}: {info['size_mb']} MB")
    
    # Summary
    total_files = len(status['files'])
    print(f"\nSummary: {total_files} files downloaded successfully")

if __name__ == "__main__":
    test_downloader() 