#!/usr/bin/env python3
"""
Targeted search for voter demographics tables with filename variations.
"""

import requests
from datetime import datetime
import time
import os

def check_url_exists(url):
    """Check if a URL exists by making a HEAD request."""
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def search_demographics_variations():
    """Search for demographics tables with different filename variations."""
    
    base_url = "https://www.sos.wa.gov/sites/default/files/{date}/{filename}"
    
    # Different filename variations to try
    filename_variations = [
        "Voter%20Demographics%20Tables.xlsx",
        "Voter Demographics Tables.xlsx",
    ]
    
    # Date variations to try (focus on recent years and common months)
    date_variations = [
        "2023-06",  # Known working date
        "2023-12", "2023-11", "2023-10", "2023-09", "2023-08", "2023-07",
        "2023-05", "2023-04", "2023-03", "2023-02", "2023-01",
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-12", "2024-11", "2024-10", "2024-09", "2024-08", "2024-07",
        "2022-12", "2022-11", "2022-10", "2022-09", "2022-08", "2022-07",
        "2022-06", "2022-05", "2022-04", "2022-03", "2022-02", "2022-01",
        "2021-12", "2021-11", "2021-10", "2021-09", "2021-08", "2021-07",
        "2021-06", "2021-05", "2021-04", "2021-03", "2021-02", "2021-01"
    ]
    
    print("Searching for Voter Demographics Tables with variations...")
    print("=" * 70)
    
    found_tables = []
    total_checks = len(filename_variations) * len(date_variations)
    current_check = 0
    
    for date_str in date_variations:
        for filename in filename_variations:
            current_check += 1
            url = base_url.format(date=date_str, filename=filename)
            
            print(f"[{current_check}/{total_checks}] Checking: {date_str}/{filename}", end=" ")
            
            if check_url_exists(url):
                print("✓ FOUND!")
                found_tables.append({
                    'date': date_str,
                    'filename': filename,
                    'url': url
                })
            else:
                print("✗")
            
            # Be respectful with requests
            time.sleep(0.3)
    
    return found_tables

def download_table(table_info, save_dir="data/raw"):
    """Download a demographics table."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename
        filename = f"wa_voter_demographics_{table_info['date']}.xlsx"
        save_path = os.path.join(save_dir, filename)
        
        print(f"Downloading {table_info['url']}...")
        response = requests.get(table_info['url'], timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded to: {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def main():
    """Main function."""
    print("Targeted Voter Demographics Table Search")
    print("=" * 50)
    print()
    
    # Search for tables
    found_tables = search_demographics_variations()
    
    if not found_tables:
        print("\nNo demographics tables found with any variations.")
        print("The current URL might be the only one available.")
        return
    
    print(f"\nFound {len(found_tables)} demographics tables:")
    print("-" * 70)
    
    for i, table in enumerate(found_tables, 1):
        print(f"{i}. {table['date']} - {table['filename']}")
        print(f"   URL: {table['url']}")
        print()
    
    # Ask if user wants to download
    print(f"Would you like to download these {len(found_tables)} tables? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        print("\nDownloading tables...")
        success_count = 0
        
        for table in found_tables:
            if download_table(table):
                success_count += 1
        
        print(f"\nDownloaded {success_count} out of {len(found_tables)} tables successfully.")
    
    # Save results to file
    import json
    with open("found_demographics_variations.json", "w") as f:
        json.dump(found_tables, f, indent=2)
    
    print(f"\nResults saved to: found_demographics_variations.json")

if __name__ == "__main__":
    main() 