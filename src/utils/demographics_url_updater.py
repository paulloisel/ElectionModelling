"""
Utility to update constants_WA.py with newly found demographics URLs.
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

def update_constants_with_demographics_urls(found_tables: List[Dict], 
                                          constants_file: str = "src/utils/constants_WA.py") -> bool:
    """
    Update the constants_WA.py file with newly found demographics URLs.
    
    Args:
        found_tables: List of dictionaries containing table information
        constants_file: Path to the constants file
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Read the current constants file
        with open(constants_file, 'r') as f:
            content = f.read()
        
        # Parse found tables to extract URLs by year
        urls_by_year = _organize_tables_by_year(found_tables)
        
        # Generate new constants content
        new_content = _generate_constants_content(content, urls_by_year)
        
        # Write the updated content back to the file
        with open(constants_file, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully updated {constants_file} with {len(found_tables)} demographics URLs")
        return True
        
    except Exception as e:
        print(f"Error updating constants file: {e}")
        return False

def _organize_tables_by_year(tables: List[Dict]) -> Dict[int, List[Dict]]:
    """Organize tables by year based on extracted dates or search years."""
    urls_by_year = {}
    
    for table in tables:
        year = None
        
        # Try to extract year from the URL date
        extracted_date = table.get('extracted_date')
        if extracted_date:
            # Look for year pattern in extracted date
            year_match = re.search(r'(\d{4})', extracted_date)
            if year_match:
                year = int(year_match.group(1))
        
        # Fall back to search year if no extracted date
        if year is None:
            year = table.get('search_year')
        
        # Skip if we still don't have a year
        if year is None:
            continue
        
        if year not in urls_by_year:
            urls_by_year[year] = []
        
        urls_by_year[year].append(table)
    
    return urls_by_year

def _generate_constants_content(original_content: str, urls_by_year: Dict[int, List[Dict]]) -> str:
    """Generate updated constants content with new demographics URLs."""
    
    # Find the current VOTER_DEMOGRAPHICS_URL line
    current_url_pattern = r'VOTER_DEMOGRAPHICS_URL\s*=\s*"[^"]*"'
    current_url_match = re.search(current_url_pattern, original_content)
    
    if not current_url_match:
        print("Could not find VOTER_DEMOGRAPHICS_URL in constants file")
        return original_content
    
    # Create new demographics URLs section
    demographics_section = _create_demographics_section(urls_by_year)
    
    # Replace the current URL with the new section
    new_content = re.sub(current_url_pattern, demographics_section, original_content)
    
    return new_content

def _create_demographics_section(urls_by_year: Dict[int, List[Dict]]) -> str:
    """Create a new demographics URLs section for the constants file."""
    
    section_lines = [
        "# Voter demographics data - Historical URLs by year",
        "VOTER_DEMOGRAPHICS_URLS = {"
    ]
    
    # Sort years in descending order
    for year in sorted(urls_by_year.keys(), reverse=True):
        section_lines.append(f"    {year}: {{")
        
        tables = urls_by_year[year]
        for i, table in enumerate(tables):
            title = table['title'].replace('"', '\\"')  # Escape quotes
            url = table['url']
            file_type = table['file_type']
            
            section_lines.append(f"        'table_{i+1}': {{")
            section_lines.append(f"            'title': \"{title}\",")
            section_lines.append(f"            'url': \"{url}\",")
            section_lines.append(f"            'file_type': '{file_type}',")
            section_lines.append(f"            'extracted_date': '{table.get('extracted_date', 'Unknown')}',")
            section_lines.append(f"            'found_date': '{table.get('found_date', 'Unknown')}'")
            section_lines.append(f"        }},")
        
        section_lines.append("    },")
    
    section_lines.append("}")
    section_lines.append("")
    section_lines.append("# Current demographics URL (most recent)")
    section_lines.append("VOTER_DEMOGRAPHICS_URL = VOTER_DEMOGRAPHICS_URLS[max(VOTER_DEMOGRAPHICS_URLS.keys())]['table_1']['url']")
    section_lines.append("VOTER_DEMOGRAPHICS_FILENAME = \"wa_voter_demographics_tables.xlsx\"")
    
    return "\n".join(section_lines)

def create_demographics_summary_report(found_tables: List[Dict], 
                                     output_file: str = "demographics_discovery_report.md") -> None:
    """
    Create a markdown report summarizing the demographics tables found.
    
    Args:
        found_tables: List of dictionaries containing table information
        output_file: Path to save the report
    """
    
    report_lines = [
        "# Washington State Voter Demographics Tables Discovery Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Tables Found:** {len(found_tables)}",
        "",
        "## Summary",
        ""
    ]
    
    # Group by year
    urls_by_year = _organize_tables_by_year(found_tables)
    
    for year in sorted(urls_by_year.keys(), reverse=True):
        report_lines.append(f"### {year} ({len(urls_by_year[year])} tables)")
        report_lines.append("")
        
        tables = urls_by_year[year]
        for i, table in enumerate(tables, 1):
            report_lines.append(f"#### Table {i}: {table['title']}")
            report_lines.append("")
            report_lines.append(f"- **URL:** {table['url']}")
            report_lines.append(f"- **File Type:** {table['file_type']}")
            report_lines.append(f"- **Extracted Date:** {table.get('extracted_date', 'Unknown')}")
            report_lines.append(f"- **Matched Term:** {table['matched_term']}")
            report_lines.append(f"- **Source Page:** {table['source_page']}")
            report_lines.append("")
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"Discovery report saved to: {output_file}")

def main():
    """Example usage of the demographics URL updater."""
    
    # Example: Load found tables from JSON file
    json_file = "demographics_tables_found.json"
    
    if not Path(json_file).exists():
        print(f"JSON file {json_file} not found. Run the scraper first.")
        return
    
    with open(json_file, 'r') as f:
        found_tables = json.load(f)
    
    print(f"Loaded {len(found_tables)} tables from {json_file}")
    
    # Create summary report
    create_demographics_summary_report(found_tables)
    
    # Update constants file
    success = update_constants_with_demographics_urls(found_tables)
    
    if success:
        print("Constants file updated successfully!")
    else:
        print("Failed to update constants file.")

if __name__ == "__main__":
    main() 