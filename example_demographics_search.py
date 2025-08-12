#!/usr/bin/env python3
"""
Comprehensive example of searching for voter demographics tables and updating constants.

This script demonstrates the full workflow:
1. Search for demographics tables on the WA SOS website
2. Display results and allow user interaction
3. Download selected tables
4. Update the constants file with new URLs
5. Generate a discovery report
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.voter_demographics_scraper import VoterDemographicsScraper
from utils.demographics_url_updater import update_constants_with_demographics_urls, create_demographics_summary_report
from datetime import datetime
import json

def main():
    """Main function demonstrating the full demographics search workflow."""
    print("Washington State Voter Demographics Discovery Tool")
    print("=" * 60)
    print()
    
    # Initialize the scraper
    scraper = VoterDemographicsScraper()
    
    # Get user preferences
    print("Search Configuration:")
    print("1. Search last 5 years (default)")
    print("2. Search last 10 years")
    print("3. Search specific years")
    print("4. Search all available years")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "2":
        current_year = datetime.now().year
        years_to_search = list(range(current_year - 10, current_year + 1))
    elif choice == "3":
        start_year = int(input("Enter start year: "))
        end_year = int(input("Enter end year: "))
        years_to_search = list(range(start_year, end_year + 1))
    elif choice == "4":
        years_to_search = None  # Will use default (last 10 years)
    else:
        current_year = datetime.now().year
        years_to_search = list(range(current_year - 5, current_year + 1))
    
    print(f"\nSearching for demographics tables...")
    if years_to_search:
        print(f"Years to search: {years_to_search}")
    else:
        print("Searching last 10 years (default)")
    
    print()
    
    try:
        # Search for demographics tables
        tables = scraper.get_historical_demographics_urls(years_to_search)
        
        if not tables:
            print("No demographics tables found.")
            return
        
        print(f"Found {len(tables)} demographics tables!")
        print("-" * 80)
        
        # Display results with more details
        for i, table in enumerate(tables, 1):
            print(f"{i:2d}. {table['title']}")
            print(f"     URL: {table['url']}")
            print(f"     Type: {table['file_type']} | Year: {table.get('search_year', 'Unknown')} | Date: {table.get('extracted_date', 'Unknown')}")
            print(f"     Matched: {table['matched_term']}")
            print()
        
        # Save results to JSON
        json_file = "demographics_tables_found.json"
        with open(json_file, 'w') as f:
            json.dump(tables, f, indent=2, default=str)
        print(f"Results saved to: {json_file}")
        
        # User interaction menu
        while True:
            print("\n" + "=" * 60)
            print("What would you like to do?")
            print("1. Download specific tables")
            print("2. Download all tables")
            print("3. Update constants file with new URLs")
            print("4. Generate discovery report")
            print("5. View table details")
            print("6. Exit")
            
            action = input("\nEnter your choice (1-6): ").strip()
            
            if action == "1":
                _download_specific_tables(scraper, tables)
            elif action == "2":
                _download_all_tables(scraper, tables)
            elif action == "3":
                _update_constants_file(tables)
            elif action == "4":
                _generate_report(tables)
            elif action == "5":
                _view_table_details(tables)
            elif action == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"Error during search: {e}")

def _download_specific_tables(scraper, tables):
    """Download specific tables selected by user."""
    print("\nEnter the numbers of tables to download (comma-separated): ", end="")
    try:
        indices = [int(x.strip()) - 1 for x in input().split(',')]
        
        success_count = 0
        for idx in indices:
            if 0 <= idx < len(tables):
                table = tables[idx]
                print(f"Downloading: {table['title']}")
                success = scraper.download_demographics_table(table)
                if success:
                    print(f"✓ Successfully downloaded: {table['title']}")
                    success_count += 1
                else:
                    print(f"✗ Failed to download: {table['title']}")
            else:
                print(f"Invalid index: {idx + 1}")
        
        print(f"\nDownloaded {success_count} out of {len(indices)} tables successfully.")
        
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")

def _download_all_tables(scraper, tables):
    """Download all found tables."""
    print(f"\nDownloading all {len(tables)} tables...")
    
    success_count = 0
    for i, table in enumerate(tables, 1):
        print(f"[{i}/{len(tables)}] Downloading: {table['title']}")
        success = scraper.download_demographics_table(table)
        if success:
            print(f"✓ Successfully downloaded: {table['title']}")
            success_count += 1
        else:
            print(f"✗ Failed to download: {table['title']}")
    
    print(f"\nDownloaded {success_count} out of {len(tables)} tables successfully.")

def _update_constants_file(tables):
    """Update the constants file with new demographics URLs."""
    print("\nUpdating constants file...")
    success = update_constants_with_demographics_urls(tables)
    
    if success:
        print("✓ Constants file updated successfully!")
        print("The VOTER_DEMOGRAPHICS_URLS dictionary has been added to constants_WA.py")
    else:
        print("✗ Failed to update constants file.")

def _generate_report(tables):
    """Generate a discovery report."""
    print("\nGenerating discovery report...")
    create_demographics_summary_report(tables)
    print("✓ Discovery report generated: demographics_discovery_report.md")

def _view_table_details(tables):
    """View detailed information about a specific table."""
    print(f"\nEnter table number (1-{len(tables)}): ", end="")
    try:
        idx = int(input().strip()) - 1
        if 0 <= idx < len(tables):
            table = tables[idx]
            print("\n" + "=" * 60)
            print(f"Table Details: {table['title']}")
            print("=" * 60)
            for key, value in table.items():
                print(f"{key:20}: {value}")
        else:
            print(f"Invalid table number. Please enter a number between 1 and {len(tables)}")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main() 