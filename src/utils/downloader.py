import requests
import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from urllib.parse import urlparse
import time
import zipfile
import io
from .constants_WA import (
    SUPPORTED_YEARS, ELECTION_DATES, SOS_BASE_URL, RESULTS_BASE_URL,
    SOS_FILENAME_PATTERN, RESULTS_FILENAME_PATTERN,
    LOCAL_SOS_FILENAME_PATTERN, LOCAL_RESULTS_FILENAME_PATTERN,
    ELECTION_TYPES, MIN_FILE_SIZE_BYTES, REQUEST_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS, USER_AGENT, REGISTERED_VOTERS_URLS, ELECTION_RESULTS_URLS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WAStateDownloader:
    """
    Downloader for Washington State election data from multiple sources.
    
    Downloads:
    1. Registered voters by district (from sos.wa.gov)
    2. Election results and ballot casts (from results.vote.wa.gov)
    """
    
    def __init__(self, output_dir: str = "data/raw/wa"):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URLs
        self.sos_base_url = SOS_BASE_URL
        self.results_base_url = RESULTS_BASE_URL
        
        # Session for requests with timeout and retry logic
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT
        })
    
    def _download_file(self, url: str, filename: str, max_retries: int = MAX_RETRY_ATTEMPTS) -> bool:
        """
        Download a file with retry logic and validation.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if download successful, False otherwise
        """
        filepath = self.output_dir / filename
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                
                # Check if we got actual content
                if len(response.content) < MIN_FILE_SIZE_BYTES:
                    logger.warning(f"Downloaded file seems too small: {len(response.content)} bytes")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                
                # Save the file
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded {filename} ({len(response.content)} bytes)")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
                    return False
        
        return False
    
    def _download_zip_file(self, url: str, target_filename: str, local_filename: str, max_retries: int = MAX_RETRY_ATTEMPTS) -> bool:
        """
        Download a ZIP file and extract a specific file from it.
        
        Args:
            url: URL to download the ZIP file from
            target_filename: Name of the file to extract from the ZIP
            local_filename: Local filename to save the extracted file as
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if download and extraction successful, False otherwise
        """
        filepath = self.output_dir / local_filename
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading ZIP file {url} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                
                # Check if we got actual content
                if len(response.content) < MIN_FILE_SIZE_BYTES:
                    logger.warning(f"Downloaded ZIP file seems too small: {len(response.content)} bytes")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                
                # Extract the target file from the ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    # Check if the target file exists in the ZIP
                    if target_filename not in zip_file.namelist():
                        logger.error(f"File '{target_filename}' not found in ZIP archive")
                        logger.info(f"Available files in ZIP: {zip_file.namelist()}")
                        return False
                    
                    # Extract the target file
                    with zip_file.open(target_filename) as zip_entry:
                        content = zip_entry.read()
                        
                        # Save the extracted file
                        with open(filepath, 'wb') as f:
                            f.write(content)
                
                logger.info(f"Successfully extracted {target_filename} from ZIP and saved as {local_filename} ({len(content)} bytes)")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
                    return False
            except zipfile.BadZipFile as e:
                logger.error(f"Invalid ZIP file: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False
            except Exception as e:
                logger.error(f"Unexpected error during ZIP processing: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False
        
        return False
    
    def _validate_file(self, filepath: Path, file_type: str) -> bool:
        """
        Validate downloaded file based on type.
        
        Args:
            filepath: Path to the downloaded file
            file_type: Type of file ('excel' or 'csv')
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            if not filepath.exists():
                logger.error(f"File does not exist: {filepath}")
                return False
            
            file_size = filepath.stat().st_size
            if file_size < MIN_FILE_SIZE_BYTES:
                logger.error(f"File too small: {file_size} bytes")
                return False
            
            # Try to read the file based on type
            if file_type == 'excel':
                df = pd.read_excel(filepath)
                if df.empty:
                    logger.error("Excel file is empty")
                    return False
                logger.info(f"Excel file validated: {len(df)} rows, {len(df.columns)} columns")
                
            elif file_type == 'csv':
                df = pd.read_csv(filepath)
                if df.empty:
                    logger.error("CSV file is empty")
                    return False
                logger.info(f"CSV file validated: {len(df)} rows, {len(df.columns)} columns")
            
            elif file_type == 'text':
                # For text files, just check if they have content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        logger.error("Text file is empty")
                        return False
                    lines = content.strip().split('\n')
                    logger.info(f"Text file validated: {len(lines)} lines")
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False
    
    def download_registered_voters(self, year: int, election_type: str = ELECTION_TYPES['primary']) -> Optional[Path]:
        """
        Download registered voters data by congressional district.
        
        Args:
            year: Election year (e.g., 2024, 2020)
            election_type: Election type ("Pri" for primary, "Gen" for general)
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        local_filename = LOCAL_SOS_FILENAME_PATTERN.format(year=year, election_type=election_type.lower())
        
        logger.info(f"Downloading registered voters data for {year} {election_type}")
        
        # Check if this year has special URL handling
        election_type_key = 'primary' if election_type == ELECTION_TYPES['primary'] else 'general'
        if year in REGISTERED_VOTERS_URLS and election_type_key in REGISTERED_VOTERS_URLS[year]:
            special_config = REGISTERED_VOTERS_URLS[year][election_type_key]
            url = special_config['url']
            is_zip = special_config['is_zip']
            target_filename = special_config['filename']
            
            if is_zip:
                # Handle ZIP file download
                if self._download_zip_file(url, target_filename, local_filename):
                    filepath = self.output_dir / local_filename
                    
                    # Determine validation type based on the actual file content
                    # For 2014 primary, it's a text file but we save it as .xlsx for consistency
                    if year == 2014 and election_type == ELECTION_TYPES['primary']:
                        validation_type = 'text'
                    else:
                        validation_type = 'excel'
                    
                    if self._validate_file(filepath, validation_type):
                        logger.info(f"Successfully downloaded and validated registered voters data from ZIP: {filepath}")
                        return filepath
                    else:
                        logger.error("File validation failed")
                        filepath.unlink(missing_ok=True)  # Clean up invalid file
                        return None
                else:
                    logger.error("ZIP download failed")
                    return None
            else:
                # Handle direct file download with special URL
                if self._download_file(url, local_filename):
                    filepath = self.output_dir / local_filename
                    if self._validate_file(filepath, 'excel'):
                        logger.info(f"Successfully downloaded and validated registered voters data: {filepath}")
                        return filepath
                    else:
                        logger.error("File validation failed")
                        filepath.unlink(missing_ok=True)  # Clean up invalid file
                        return None
                else:
                    logger.error("Download failed")
                    return None
        else:
            # Use standard URL pattern
            filename = SOS_FILENAME_PATTERN.format(year=year, election_type=election_type)
            url = f"{self.sos_base_url}/{year}-09/{filename}"
            
            if self._download_file(url, local_filename):
                filepath = self.output_dir / local_filename
                if self._validate_file(filepath, 'excel'):
                    logger.info(f"Successfully downloaded and validated registered voters data: {filepath}")
                    return filepath
                else:
                    logger.error("File validation failed")
                    filepath.unlink(missing_ok=True)  # Clean up invalid file
                    return None
            else:
                logger.error("Download failed")
                return None
    
    def download_election_results(self, year: int, month: int, day: int) -> Optional[Path]:
        """
        Download election results and ballot casts data.
        
        Args:
            year: Election year (e.g., 2024)
            month: Election month (e.g., 8 for August)
            day: Election day (e.g., 6)
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        # Format date as YYYYMMDD
        date_str = f"{year}{month:02d}{day:02d}"
        
        # Check if this year has special URL handling for election results
        if year in ELECTION_RESULTS_URLS:
            # Determine election type based on date
            if (month, day) == ELECTION_DATES[year]['primary']:
                election_type = 'primary'
            elif (month, day) == ELECTION_DATES[year]['general']:
                election_type = 'general'
            else:
                logger.error(f"Unknown election date for {year}: {month}/{day}")
                return None
            
            if election_type in ELECTION_RESULTS_URLS[year]:
                special_config = ELECTION_RESULTS_URLS[year][election_type]
                url = special_config['url']
                local_filename = special_config['local_filename']
                
                logger.info(f"Downloading election results for {year} {election_type} using special URL")
                
                if self._download_file(url, local_filename):
                    filepath = self.output_dir / local_filename
                    if self._validate_file(filepath, 'csv'):
                        logger.info(f"Successfully downloaded and validated election results: {filepath}")
                        return filepath
                    else:
                        logger.error("File validation failed")
                        filepath.unlink(missing_ok=True)  # Clean up invalid file
                        return None
                else:
                    logger.error("Download failed")
                    return None
        
        # Use standard URL pattern
        filename = RESULTS_FILENAME_PATTERN.format(date_str=date_str)
        url = f"{self.results_base_url}/{date_str}/export/{filename}"
        
        local_filename = LOCAL_RESULTS_FILENAME_PATTERN.format(date_str=date_str)
        
        logger.info(f"Downloading election results for {date_str}")
        
        if self._download_file(url, local_filename):
            filepath = self.output_dir / local_filename
            if self._validate_file(filepath, 'csv'):
                logger.info(f"Successfully downloaded and validated election results: {filepath}")
                return filepath
            else:
                logger.error("File validation failed")
                filepath.unlink(missing_ok=True)  # Clean up invalid file
                return None
        else:
            logger.error("Download failed")
            return None
    
    def download_2024_primary_data(self) -> Dict[str, Optional[Path]]:
        """
        Download 2024 primary election data (both registered voters and results).
        
        Returns:
            Dictionary with paths to downloaded files
        """
        logger.info("Downloading 2024 primary election data")
        
        results = {}
        
        # Download registered voters data
        registered_voters_path = self.download_registered_voters(2024, ELECTION_TYPES['primary'])
        results['registered_voters'] = registered_voters_path
        
        # Download election results (August 6, 2024)
        primary_month, primary_day = ELECTION_DATES[2024]['primary']
        election_results_path = self.download_election_results(2024, primary_month, primary_day)
        results['election_results'] = election_results_path
        
        return results
    
    def download_2020_general_data(self) -> Dict[str, Optional[Path]]:
        """
        Download 2020 general election data (both registered voters and results).
        
        Returns:
            Dictionary with paths to downloaded files
        """
        logger.info("Downloading 2020 general election data")
        
        results = {}
        
        # Download registered voters data
        registered_voters_path = self.download_registered_voters(2020, ELECTION_TYPES['general'])
        results['registered_voters'] = registered_voters_path
        
        # Download election results (November 3, 2020)
        general_month, general_day = ELECTION_DATES[2020]['general']
        election_results_path = self.download_election_results(2020, general_month, general_day)
        results['election_results'] = election_results_path
        
        return results
    
    def download_all_even_years_data(self) -> Dict[int, Dict[str, Optional[Path]]]:
        """
        Download data for all even years between 2012 and 2024.
        
        Returns:
            Dictionary with year as key and download results as value
        """
        all_results = {}
        
        for year in SUPPORTED_YEARS:
            logger.info(f"Processing year {year}")
            year_results = {}
            
            # Download registered voters data for both primary and general elections
            primary_voters = self.download_registered_voters(year, ELECTION_TYPES['primary'])
            general_voters = self.download_registered_voters(year, ELECTION_TYPES['general'])
            
            year_results['primary_registered_voters'] = primary_voters
            year_results['general_registered_voters'] = general_voters
            
            # Download election results based on year
            if year in ELECTION_DATES:
                primary_month, primary_day = ELECTION_DATES[year]['primary']
                general_month, general_day = ELECTION_DATES[year]['general']
                
                primary_results = self.download_election_results(year, primary_month, primary_day)
                general_results = self.download_election_results(year, general_month, general_day)
                
                year_results['primary_election_results'] = primary_results
                year_results['general_election_results'] = general_results
            else:
                logger.warning(f"No election dates found for year {year}")
                year_results['primary_election_results'] = None
                year_results['general_election_results'] = None
            
            all_results[year] = year_results
            
            logger.info(f"Completed processing year {year}")
        
        return all_results
    
    def download_specific_year_data(self, year: int) -> Dict[str, Optional[Path]]:
        """
        Download data for a specific year (both primary and general elections).
        
        Args:
            year: Year to download data for (must be even year between 2012-2024)
            
        Returns:
            Dictionary with paths to downloaded files
        """
        if year not in SUPPORTED_YEARS:
            raise ValueError(f"Year {year} not supported. Must be one of {SUPPORTED_YEARS}")
        
        logger.info(f"Downloading data for year {year}")
        
        results = {}
        
        # Download registered voters data for both primary and general elections
        primary_voters = self.download_registered_voters(year, ELECTION_TYPES['primary'])
        general_voters = self.download_registered_voters(year, ELECTION_TYPES['general'])
        
        results['primary_registered_voters'] = primary_voters
        results['general_registered_voters'] = general_voters
        
        # Download election results based on year
        if year in ELECTION_DATES:
            primary_month, primary_day = ELECTION_DATES[year]['primary']
            general_month, general_day = ELECTION_DATES[year]['general']
            
            primary_results = self.download_election_results(year, primary_month, primary_day)
            general_results = self.download_election_results(year, general_month, general_day)
            
            results['primary_election_results'] = primary_results
            results['general_election_results'] = general_results
        
        return results
    
    def get_download_status(self) -> Dict[str, Any]:
        """
        Get status of downloaded files.
        
        Returns:
            Dictionary with file status information
        """
        status = {
            'output_directory': str(self.output_dir),
            'files': {}
        }
        
        for filepath in self.output_dir.glob("*"):
            if filepath.is_file():
                file_info = {
                    'size_bytes': filepath.stat().st_size,
                    'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
                    'modified': filepath.stat().st_mtime
                }
                status['files'][filepath.name] = file_info
        
        return status


def main():
    """
    Example usage of the WAStateDownloader.
    Downloads data for all even years between 2012 and 2024.
    """
    # Initialize downloader with data/raw directory
    downloader = WAStateDownloader(output_dir="data/raw")
    
    print("Washington State Election Data Downloader")
    print("=" * 50)
    
    # Download data for all even years between 2012 and 2024
    print("Downloading data for all even years (2012-2024)...")
    all_results = downloader.download_all_even_years_data()
    
    # Print summary of results
    print("\nDownload Summary:")
    print("-" * 30)
    
    total_successful = 0
    total_attempted = 0
    
    for year, year_results in all_results.items():
        print(f"\n{year}:")
        year_successful = 0
        year_attempted = 0
        
        for data_type, filepath in year_results.items():
            total_attempted += 1
            year_attempted += 1
            
            if filepath:
                status = "✓ SUCCESS"
                total_successful += 1
                year_successful += 1
                print(f"  {data_type}: {status} -> {filepath.name}")
            else:
                status = "✗ FAILED"
                print(f"  {data_type}: {status}")
        
        print(f"  Year {year} summary: {year_successful}/{year_attempted} successful")
    
    # Print overall status
    print(f"\nOverall Summary: {total_successful}/{total_attempted} downloads successful")
    
    # Print file status
    print("\nDownload Status:")
    status = downloader.get_download_status()
    print(f"Output directory: {status['output_directory']}")
    print("Files downloaded:")
    for filename, info in status['files'].items():
        print(f"  {filename}: {info['size_mb']} MB")
    
    print(f"\nTotal files: {len(status['files'])}")


if __name__ == "__main__":
    main()
