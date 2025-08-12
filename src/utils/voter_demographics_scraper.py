"""
Voter Demographics Table Scraper for Washington State

This module provides functionality to search for voter demographics tables
on the Washington Secretary of State website for different dates.
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin, urlparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoterDemographicsScraper:
    """Scraper for finding voter demographics tables on WA SOS website."""
    
    def __init__(self):
        self.base_url = "https://www.sos.wa.gov"
        self.search_url = "https://www.sos.wa.gov/elections/research/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_demographics_tables(self, 
                                 search_terms: Optional[List[str]] = None,
                                 date_range: Optional[tuple] = None) -> List[Dict]:
        """
        Search for voter demographics tables on the WA SOS website.
        
        Args:
            search_terms: List of search terms to look for (default: demographics-related terms)
            date_range: Tuple of (start_date, end_date) as datetime objects
            
        Returns:
            List of dictionaries containing found table information
        """
        if search_terms is None:
            search_terms = [
                "voter demographics",
                "demographics table",
                "voter registration demographics",
                "demographic data",
                "voter statistics",
                "registration statistics"
            ]
        
        found_tables = []
        
        # Search the main research page
        logger.info("Searching main research page...")
        main_page_tables = self._search_main_page(search_terms)
        found_tables.extend(main_page_tables)
        
        # Search the files directory
        logger.info("Searching files directory...")
        files_tables = self._search_files_directory(search_terms, date_range)
        found_tables.extend(files_tables)
        
        # Remove duplicates based on URL
        unique_tables = self._remove_duplicates(found_tables)
        
        return unique_tables
    
    def _search_main_page(self, search_terms: List[str]) -> List[Dict]:
        """Search the main research page for demographics tables."""
        try:
            response = self.session.get(self.search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            found_tables = []
            
            # Look for links that might contain demographics data
            links = soup.find_all('a', href=True)
            
            for link in links:
                link_text = link.get_text().lower()
                href = link.get('href', '')
                
                # Check if link text or href contains any search terms
                for term in search_terms:
                    if term.lower() in link_text or term.lower() in href.lower():
                        # Check if it's an Excel file or PDF
                        if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv', '.pdf']):
                            table_info = {
                                'title': link.get_text().strip(),
                                'url': urljoin(self.base_url, href),
                                'source_page': self.search_url,
                                'found_date': datetime.now().isoformat(),
                                'file_type': self._get_file_type(href),
                                'matched_term': term
                            }
                            found_tables.append(table_info)
                            logger.info(f"Found table: {table_info['title']}")
            
            return found_tables
            
        except Exception as e:
            logger.error(f"Error searching main page: {e}")
            return []
    
    def _search_files_directory(self, search_terms: List[str], date_range: Optional[tuple]) -> List[Dict]:
        """Search the files directory for demographics tables."""
        files_url = "https://www.sos.wa.gov/sites/default/files"
        
        try:
            response = self.session.get(files_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            found_tables = []
            
            # Look for directory listings or links
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                link_text = link.get_text().lower()
                
                # Check if it's a demographics-related file
                for term in search_terms:
                    if term.lower() in link_text or term.lower() in href.lower():
                        if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv', '.pdf']):
                            # Check date range if specified
                            if date_range and not self._is_in_date_range(href, date_range):
                                continue
                                
                            table_info = {
                                'title': link.get_text().strip(),
                                'url': urljoin(files_url, href),
                                'source_page': files_url,
                                'found_date': datetime.now().isoformat(),
                                'file_type': self._get_file_type(href),
                                'matched_term': term,
                                'extracted_date': self._extract_date_from_url(href)
                            }
                            found_tables.append(table_info)
                            logger.info(f"Found table in files: {table_info['title']}")
            
            return found_tables
            
        except Exception as e:
            logger.error(f"Error searching files directory: {e}")
            return []
    
    def _get_file_type(self, url: str) -> str:
        """Extract file type from URL."""
        url_lower = url.lower()
        if '.xlsx' in url_lower:
            return 'xlsx'
        elif '.xls' in url_lower:
            return 'xls'
        elif '.csv' in url_lower:
            return 'csv'
        elif '.pdf' in url_lower:
            return 'pdf'
        else:
            return 'unknown'
    
    def _extract_date_from_url(self, url: str) -> Optional[str]:
        """Extract date from URL if present."""
        # Look for date patterns in URL
        date_patterns = [
            r'(\d{4}-\d{2})',  # YYYY-MM
            r'(\d{4})',        # YYYY
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _is_in_date_range(self, url: str, date_range: tuple) -> bool:
        """Check if URL contains a date within the specified range."""
        extracted_date = self._extract_date_from_url(url)
        if not extracted_date:
            return True  # If no date found, include it
        
        try:
            # Try to parse the extracted date
            if len(extracted_date) == 4:  # YYYY
                url_date = datetime.strptime(extracted_date, '%Y')
            elif len(extracted_date) == 7:  # YYYY-MM
                url_date = datetime.strptime(extracted_date, '%Y-%m')
            else:
                return True  # If can't parse, include it
            
            start_date, end_date = date_range
            return start_date <= url_date <= end_date
            
        except ValueError:
            return True  # If parsing fails, include it
    
    def _remove_duplicates(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables based on URL."""
        seen_urls = set()
        unique_tables = []
        
        for table in tables:
            if table['url'] not in seen_urls:
                seen_urls.add(table['url'])
                unique_tables.append(table)
        
        return unique_tables
    
    def get_historical_demographics_urls(self, years: List[int] = None) -> List[Dict]:
        """
        Get URLs for historical voter demographics tables.
        
        Args:
            years: List of years to search for (default: last 10 years)
            
        Returns:
            List of dictionaries with demographics table information
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 10, current_year + 1))
        
        all_tables = []
        
        for year in years:
            logger.info(f"Searching for demographics tables from {year}...")
            
            # Create date range for the year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            # Enhanced year-specific search terms
            year_terms = [
                f"{year} voter demographics",
                f"{year} demographics table",
                f"voter demographics {year}",
                f"{year} demographics",
                f"demographics {year}",
                "demographics table",
                "voter demographics table",
                "demographics tables"
            ]
            
            year_tables = self.search_demographics_tables(
                search_terms=year_terms,
                date_range=(start_date, end_date)
            )
            
            # Add year information to results
            for table in year_tables:
                table['search_year'] = year
            
            all_tables.extend(year_tables)
            
            # Be respectful with requests
            time.sleep(1)
        
        return all_tables
    
    def search_demographics_by_pattern(self, base_url: str, year_range: tuple) -> List[Dict]:
        """
        Search for demographics tables using URL pattern matching.
        
        Args:
            base_url: Base URL pattern to search
            year_range: Tuple of (start_year, end_year)
            
        Returns:
            List of dictionaries with demographics table information
        """
        start_year, end_year = year_range
        found_tables = []
        
        # Common URL patterns for demographics tables
        url_patterns = [
            "{year}-{month:02d}/Voter%20Demographics%20Tables.xlsx",
            "{year}-{month:02d}/voter%20demographics%20tables.xlsx",
            "{year}-{month:02d}/Demographics%20Tables.xlsx",
            "{year}-{month:02d}/demographics%20tables.xlsx",
            "{year}-{month:02d}/Voter%20Demographics.xlsx",
            "{year}-{month:02d}/voter%20demographics.xlsx"
        ]
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                for pattern in url_patterns:
                    try:
                        url = base_url + pattern.format(year=year, month=month)
                        logger.info(f"Trying URL pattern: {url}")
                        
                        response = self.session.head(url, timeout=10)
                        if response.status_code == 200:
                            table_info = {
                                'title': f"Voter Demographics Tables {year}-{month:02d}",
                                'url': url,
                                'source_page': 'URL Pattern Search',
                                'found_date': datetime.now().isoformat(),
                                'file_type': 'xlsx',
                                'matched_term': 'URL Pattern',
                                'search_year': year,
                                'extracted_date': f"{year}-{month:02d}"
                            }
                            found_tables.append(table_info)
                            logger.info(f"Found table via pattern: {url}")
                        
                        # Be respectful with requests
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.debug(f"Pattern {pattern} failed for {year}-{month}: {e}")
                        continue
        
        return found_tables
    
    def download_demographics_table(self, table_info: Dict, save_path: str = None) -> bool:
        """
        Download a demographics table from the provided URL.
        
        Args:
            table_info: Dictionary containing table information
            save_path: Path to save the file (default: use filename from URL)
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            url = table_info['url']
            
            if save_path is None:
                # Extract filename from URL
                parsed_url = urlparse(url)
                filename = parsed_url.path.split('/')[-1]
                save_path = f"data/raw/{filename}"
            
            logger.info(f"Downloading {url} to {save_path}...")
            
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading table: {e}")
            return False


def main():
    """Example usage of the VoterDemographicsScraper."""
    scraper = VoterDemographicsScraper()
    
    # Search for demographics tables from the last 5 years
    current_year = datetime.now().year
    years_to_search = list(range(current_year - 5, current_year + 1))
    
    print("Searching for voter demographics tables...")
    tables = scraper.get_historical_demographics_urls(years_to_search)
    
    print(f"\nFound {len(tables)} demographics tables:")
    print("-" * 80)
    
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table['title']}")
        print(f"   URL: {table['url']}")
        print(f"   Type: {table['file_type']}")
        print(f"   Year: {table.get('search_year', 'Unknown')}")
        print(f"   Date: {table.get('extracted_date', 'Unknown')}")
        print(f"   Matched: {table['matched_term']}")
        print()


if __name__ == "__main__":
    main() 