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
    MAX_RETRY_ATTEMPTS, USER_AGENT, REGISTERED_VOTERS_URLS, ELECTION_RESULTS_URLS, GEOGRAPHIC_DATA_2012,
    CENSUS_YEARS, CENSUS_VARIABLES, CENSUS_BASE_URL, WA_STATE_FIPS, WA_CENSUS_GEO,
    VOTER_DEMOGRAPHICS_URL, VOTER_DEMOGRAPHICS_FILENAME
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
    
    def download_wa_congressional_districts_2012_redistricting(self) -> Optional[Path]:
        """
        Download Washington congressional district boundaries (Post-2010 redistricting).
        
        Downloads the TIGER/Line shapefile for Washington congressional districts
        (113th-115th Congresses) from the US Census Bureau.
        This represents the 2012 redistricting boundaries that were in effect
        for the 113th-115th Congresses.
        
        Returns:
            Path to downloaded ZIP file if successful, None otherwise
        """
        # Get constants for 2012 geographic data
        url = GEOGRAPHIC_DATA_2012['wa_congressional_districts_url']
        local_filename = GEOGRAPHIC_DATA_2012['wa_congressional_districts_filename']
        description = GEOGRAPHIC_DATA_2012['description']
        
        # Create geographics subdirectory if it doesn't exist
        geographics_dir = self.output_dir / "geographics"
        geographics_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = geographics_dir / local_filename
        
        logger.info(f"Downloading Washington congressional district boundaries ({description}) from {url}")
        
        # Use the existing _download_file method
        if self._download_file(url, f"geographics/{local_filename}"):
            logger.info(f"Successfully downloaded congressional district boundaries: {filepath}")
            return filepath
        else:
            logger.error("Failed to download congressional district boundaries")
            return None
    
    def download_voter_demographics(self) -> Optional[Path]:
        """
        Download Washington State voter demographics data.
        
        Downloads the voter demographics tables Excel file from the Washington State
        Secretary of State website. This file contains demographic information about
        registered voters in Washington State.
        
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        logger.info(f"Downloading Washington State voter demographics data from {VOTER_DEMOGRAPHICS_URL}")
        
        # Create demographics subdirectory directly in data/raw
        demographics_dir = Path("data/raw/demographics")
        demographics_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = demographics_dir / VOTER_DEMOGRAPHICS_FILENAME
        
        # Download the file directly to the demographics directory
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                logger.info(f"Downloading {VOTER_DEMOGRAPHICS_URL} (attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                
                response = self.session.get(VOTER_DEMOGRAPHICS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                
                # Check if we got actual content
                if len(response.content) < MIN_FILE_SIZE_BYTES:
                    logger.warning(f"Downloaded file seems too small: {len(response.content)} bytes")
                    if attempt < MAX_RETRY_ATTEMPTS - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                
                # Save the file
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded {VOTER_DEMOGRAPHICS_FILENAME} ({len(response.content)} bytes)")
                
                # Validate the downloaded file
                if self._validate_file(filepath, 'excel'):
                    logger.info(f"Successfully downloaded and validated voter demographics data: {filepath}")
                    return filepath
                else:
                    logger.error("Voter demographics file validation failed")
                    filepath.unlink(missing_ok=True)  # Clean up invalid file
                    return None
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {VOTER_DEMOGRAPHICS_URL} after {MAX_RETRY_ATTEMPTS} attempts")
                    return None
        
        return None
    
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


class CensusDownloader:
    """
    Generic downloader for US Census data.
    
    Downloads American Community Survey (ACS) 5-year estimates for any state
    and geographic level from 2010-2023.
    """
    
    def __init__(self, output_dir: str = "data/raw/census"):
        """
        Initialize the census downloader.
        
        Args:
            output_dir: Directory to save downloaded census files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Census-specific attributes
        self.years = CENSUS_YEARS
        self.variables = CENSUS_VARIABLES
        self.base_url = CENSUS_BASE_URL
    
    def download_census_data(self, year: int, geo: list, state_name: str = "state", 
                           variables: list = None, geographic_level: str = "congressional district") -> Optional[pd.DataFrame]:
        """
        Download census data for a specific year and geographic level.
        
        Args:
            year: Census year (2010-2023)
            geo: Geographic specification for censusdata.censusgeo()
            state_name: Name of the state (for file naming)
            variables: List of census variables to download (defaults to CENSUS_VARIABLES)
            geographic_level: Geographic level (e.g., "congressional district", "county", etc.)
            
        Returns:
            DataFrame with census data if successful, None otherwise
        """
        if variables is None:
            variables = self.variables
        
        if year not in self.years:
            raise ValueError(f"Year {year} not supported. Must be one of {list(self.years)}")
        
        try:
            import censusdata
            
            logger.info(f"Downloading census data for {state_name} {geographic_level}s, year {year}")
            
            # Download data
            df = censusdata.download("acs5", year, geo, variables)
            
            # Reset index to convert index to columns
            df.reset_index(inplace=True)
            
            # Rename the index column to "NAME"
            df.rename(columns={"index": "NAME"}, inplace=True)
            
            # Add year column
            df["year"] = year
            
            # Save to file
            filename = f"{state_name.lower()}_census_{geographic_level.replace(' ', '_')}_{year}.csv"
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            
            logger.info(f"Successfully downloaded and saved census data for {year}: {filepath}")
            logger.info(f"Data shape: {df.shape}")
            
            return df
            
        except ImportError:
            logger.error("censusdata package not installed. Please install it with: pip install censusdata")
            return None
        except Exception as e:
            logger.error(f"Failed to download census data for year {year}: {e}")
            return None
    
    def download_wa_congressional_districts(self, year: int, variables: list = None) -> Optional[pd.DataFrame]:
        """
        Download census data for Washington State congressional districts for a specific year.
        
        Args:
            year: Census year (2010-2023)
            variables: List of census variables to download (defaults to CENSUS_VARIABLES)
            
        Returns:
            DataFrame with census data if successful, None otherwise
        """
        try:
            import censusdata
            geo = censusdata.censusgeo(WA_CENSUS_GEO)
        except ImportError:
            logger.error("censusdata package not installed. Please install it with: pip install censusdata")
            return None
        
        return self.download_census_data(
            year=year,
            geo=geo,
            state_name="wa",
            variables=variables,
            geographic_level="congressional district"
        )
    
    def download_all_wa_congressional_districts(self) -> Dict[int, Optional[pd.DataFrame]]:
        """
        Download census data for Washington State congressional districts for all years (2010-2023).
        
        Returns:
            Dictionary with year as key and DataFrame as value
        """
        logger.info("Downloading census data for Washington State congressional districts for all years (2010-2023)")
        
        all_data = {}
        
        for year in self.years:
            logger.info(f"Processing census data for year {year}")
            df = self.download_wa_congressional_districts(year)
            all_data[year] = df
            
            # Add a small delay between requests to be respectful to the API
            time.sleep(1)
        
        return all_data
    
    def download_and_combine_wa_congressional_districts(self) -> Optional[pd.DataFrame]:
        """
        Download census data for Washington State congressional districts for all years and combine into a single DataFrame.
        
        Returns:
            Combined DataFrame with all years of census data, None if failed
        """
        logger.info("Downloading and combining Washington State congressional district census data for all years")
        
        try:
            # Download data for all years
            all_data = self.download_all_wa_congressional_districts()
            
            # Filter out None values (failed downloads)
            successful_data = {year: df for year, df in all_data.items() if df is not None}
            
            if not successful_data:
                logger.error("No census data was successfully downloaded")
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(successful_data.values(), ignore_index=True)
            
            # Save combined data
            combined_filename = "wa_census_congressional_districts_combined_2010_2023.csv"
            combined_filepath = self.output_dir / combined_filename
            combined_df.to_csv(combined_filepath, index=False)
            
            logger.info(f"Successfully combined census data: {combined_filepath}")
            logger.info(f"Combined data shape: {combined_df.shape}")
            logger.info(f"Years included: {sorted(successful_data.keys())}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to combine census data: {e}")
            return None
    
    def get_download_status(self) -> Dict[str, Any]:
        """
        Get status of downloaded census files.
        
        Returns:
            Dictionary with census file status information
        """
        status = {
            'census_directory': str(self.output_dir),
            'files': {}
        }
        
        if self.output_dir.exists():
            for filepath in self.output_dir.glob("*.csv"):
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
    
    # Download congressional district boundaries
    print("Downloading Washington congressional district boundaries (2012 redistricting)...")
    districts_path = downloader.download_wa_congressional_districts_2012_redistricting()
    if districts_path:
        print(f"✓ Congressional districts downloaded: {districts_path}")
    else:
        print("✗ Failed to download congressional districts")
    
    # Download voter demographics data
    print("Downloading Washington State voter demographics data...")
    demographics_path = downloader.download_voter_demographics()
    if demographics_path:
        print(f"✓ Voter demographics downloaded: {demographics_path}")
    else:
        print("✗ Failed to download voter demographics")
    
    # Download data for all even years between 2012 and 2024
    print("\nDownloading data for all even years (2012-2024)...")
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
    
    # Example of using the census downloader
    print("\n" + "=" * 50)
    print("Census Data Download Example")
    print("=" * 50)
    
    # Initialize census downloader
    census_downloader = CensusDownloader(output_dir="data/raw/census")
    
    # Download and combine census data for all years
    print("Downloading census data for Washington State congressional districts (2010-2023)...")
    combined_census_data = census_downloader.download_and_combine_wa_congressional_districts()
    
    if combined_census_data is not None:
        print(f"✓ Census data downloaded and combined successfully")
        print(f"  Shape: {combined_census_data.shape}")
        print(f"  Years: {sorted(combined_census_data['year'].unique())}")
        print(f"  Congressional Districts: {sorted(combined_census_data['NAME'].unique())}")
    else:
        print("✗ Failed to download census data")
    
    # Print census download status
    print("\nCensus Download Status:")
    census_status = census_downloader.get_download_status()
    print(f"Census directory: {census_status['census_directory']}")
    print("Census files downloaded:")
    for filename, info in census_status['files'].items():
        print(f"  {filename}: {info['size_mb']} MB")


if __name__ == "__main__":
    main()
