#!/usr/bin/env python3
"""
Script to search for voter demographics tables on the Washington Secretary of State website.

This script searches for historical voter demographics tables between 2012 and 2022,
using the VoterDemographicsScraper to find similar tables to the current one.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.voter_demographics_scraper import VoterDemographicsScraper
from datetime import datetime
import json

def main():
    """Main function to search for demographics tables between 2012-2022."""
    print("Washington State Voter Demographics Table Search (2012-2022)")
    print("=" * 60)
    
    # Initialize the scraper
    scraper = VoterDemographicsScraper()
    
    # Search for tables specifically between 2012 and 2022
    years_to_search = list(range(2012, 2023))  # 2012 to 2022 inclusive
    print(f"Searching for demographics tables from years: {years_to_search}")
    print()
    
    try:
        # Enhanced search terms specifically for demographics tables
        demographics_search_terms = [
            "voter demographics",
            "demographics table", 
            "voter registration demographics",
            "demographic data",
            "voter statistics",
            "registration statistics",
            "demographics.xlsx",
            "demographics.xls",
            "voter demographics table",
            "demographics tables",
            "voter demographics tables"
        ]
        
        # Search for demographics tables with enhanced terms
        print("Searching with demographics-specific terms...")
        tables = scraper.search_demographics_tables(
            search_terms=demographics_search_terms,
            date_range=(datetime(2012, 1, 1), datetime(2022, 12, 31))
        )
        
        # Also search by years for historical data
        print("Searching for historical demographics tables by year...")
        historical_tables = scraper.get_historical_demographics_urls(years_to_search)
        
        # Search using URL patterns based on the current demographics URL
        print("Searching using URL patterns...")
        base_url = "https://www.sos.wa.gov/sites/default/files/"
        pattern_tables = scraper.search_demographics_by_pattern(base_url, (2012, 2022))
        
        # Combine and deduplicate results
        all_tables = tables + historical_tables + pattern_tables
        unique_tables = scraper._remove_duplicates(all_tables)
        
        # Filter to only include tables from 2012-2022
        filtered_tables = []
        for table in unique_tables:
            # Check if table has year information
            if 'search_year' in table:
                if 2012 <= table['search_year'] <= 2022:
                    filtered_tables.append(table)
            else:
                # Check extracted date from URL
                extracted_date = table.get('extracted_date')
                if extracted_date:
                    try:
                        if len(extracted_date) == 4:  # YYYY
                            year = int(extracted_date)
                            if 2012 <= year <= 2022:
                                filtered_tables.append(table)
                        elif len(extracted_date) == 7:  # YYYY-MM
                            year = int(extracted_date[:4])
                            if 2012 <= year <= 2022:
                                filtered_tables.append(table)
                    except (ValueError, TypeError):
                        # If we can't parse the date, include it for manual review
                        filtered_tables.append(table)
                else:
                    # If no date info, include it for manual review
                    filtered_tables.append(table)
        
        if not filtered_tables:
            print("No demographics tables found for the period 2012-2022.")
            print("\nTrying alternative search approach...")
            
            # Try searching with more generic terms
            generic_terms = [
                "demographics",
                "voter",
                "registration",
                "statistics",
                "data"
            ]
            
            generic_tables = scraper.search_demographics_tables(
                search_terms=generic_terms,
                date_range=(datetime(2012, 1, 1), datetime(2022, 12, 31))
            )
            
            if generic_tables:
                print(f"Found {len(generic_tables)} potential tables with generic search:")
                filtered_tables = generic_tables
            else:
                print("No tables found with any search method.")
                return
        else:
            print(f"Found {len(filtered_tables)} demographics tables for 2012-2022:")
        
        print("-" * 80)
        
        # Display results with enhanced information
        for i, table in enumerate(filtered_tables, 1):
            print(f"{i}. {table['title']}")
            print(f"   URL: {table['url']}")
            print(f"   File Type: {table['file_type']}")
            print(f"   Search Year: {table.get('search_year', 'Unknown')}")
            print(f"   Extracted Date: {table.get('extracted_date', 'Unknown')}")
            print(f"   Matched Term: {table['matched_term']}")
            print(f"   Source: {table['source_page']}")
            print()
        
        # Ask user if they want to download any tables
        print("Would you like to download any of these tables? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            print("\nEnter the numbers of tables to download (comma-separated): ", end="")
            try:
                indices = [int(x.strip()) - 1 for x in input().split(',')]
                
                for idx in indices:
                    if 0 <= idx < len(filtered_tables):
                        table = filtered_tables[idx]
                        success = scraper.download_demographics_table(table)
                        if success:
                            print(f"✓ Successfully downloaded: {table['title']}")
                        else:
                            print(f"✗ Failed to download: {table['title']}")
                    else:
                        print(f"Invalid index: {idx + 1}")
                        
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
        
        # Save results to JSON file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"demographics_tables_2012_2022_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(filtered_tables, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also save a summary file
        summary_file = f"demographics_summary_2012_2022_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("Washington State Voter Demographics Tables (2012-2022)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Search completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tables found: {len(filtered_tables)}\n\n")
            
            for i, table in enumerate(filtered_tables, 1):
                f.write(f"{i}. {table['title']}\n")
                f.write(f"   URL: {table['url']}\n")
                f.write(f"   Year: {table.get('search_year', table.get('extracted_date', 'Unknown'))}\n")
                f.write(f"   Type: {table['file_type']}\n\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 