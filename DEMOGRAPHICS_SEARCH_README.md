# Washington State Voter Demographics Table Search

This project includes tools to search for and discover voter demographics tables on the Washington Secretary of State website for different dates.

## Overview

The current `VOTER_DEMOGRAPHICS_URL` in `constants_WA.py` points to a table from June 2023. These tools help you:

1. **Search** for similar demographics tables from other dates
2. **Download** found tables automatically
3. **Update** the constants file with newly discovered URLs
4. **Generate** reports of discovered tables

## Files Created

### Core Modules
- `src/utils/voter_demographics_scraper.py` - Main scraper class for finding demographics tables
- `src/utils/demographics_url_updater.py` - Utility to update constants file with new URLs

### Example Scripts
- `search_demographics_tables.py` - Simple search and download script
- `example_demographics_search.py` - Comprehensive interactive workflow

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The scraper requires `beautifulsoup4` which has been added to `requirements.txt`.

## Usage

### Quick Start

Run the comprehensive example script:
```bash
python example_demographics_search.py
```

This will:
- Search for demographics tables from the last 5 years (configurable)
- Display all found tables
- Allow you to download specific tables
- Update the constants file with new URLs
- Generate a discovery report

### Simple Search

For a basic search and download:
```bash
python search_demographics_tables.py
```

### Programmatic Usage

```python
from src.utils.voter_demographics_scraper import VoterDemographicsScraper
from src.utils.demographics_url_updater import update_constants_with_demographics_urls

# Initialize scraper
scraper = VoterDemographicsScraper()

# Search for tables from specific years
years = [2020, 2021, 2022, 2023, 2024]
tables = scraper.get_historical_demographics_urls(years)

# Download a specific table
if tables:
    success = scraper.download_demographics_table(tables[0])

# Update constants file
update_constants_with_demographics_urls(tables)
```

## Features

### VoterDemographicsScraper Class

- **Search Methods**: Search main research page and files directory
- **Date Filtering**: Filter results by date range
- **File Type Detection**: Automatically detect Excel, CSV, PDF files
- **Duplicate Removal**: Remove duplicate URLs
- **Download Functionality**: Download tables to local storage

### Search Capabilities

The scraper searches for files matching these terms:
- "voter demographics"
- "demographics table"
- "voter registration demographics"
- "demographic data"
- "voter statistics"
- "registration statistics"

### Date Extraction

The scraper can extract dates from URLs using patterns:
- `YYYY-MM` (e.g., 2023-06)
- `YYYY` (e.g., 2023)
- `MM/DD/YYYY` (e.g., 06/15/2023)

### Output Files

- `demographics_tables_found.json` - Raw search results
- `demographics_discovery_report.md` - Formatted discovery report
- Downloaded files in `data/raw/` directory

## Constants File Updates

When new demographics tables are found, the constants file can be updated to include:

```python
# Voter demographics data - Historical URLs by year
VOTER_DEMOGRAPHICS_URLS = {
    2024: {
        'table_1': {
            'title': "Voter Demographics Tables",
            'url': "https://www.sos.wa.gov/sites/default/files/2024-01/Voter%20Demographics%20Tables.xlsx",
            'file_type': 'xlsx',
            'extracted_date': '2024-01',
            'found_date': '2024-01-15T10:30:00'
        },
    },
    2023: {
        'table_1': {
            'title': "Voter Demographics Tables",
            'url': "https://www.sos.wa.gov/sites/default/files/2023-06/Voter%20Demographics%20Tables.xlsx",
            'file_type': 'xlsx',
            'extracted_date': '2023-06',
            'found_date': '2024-01-15T10:30:00'
        },
    },
}

# Current demographics URL (most recent)
VOTER_DEMOGRAPHICS_URL = VOTER_DEMOGRAPHICS_URLS[max(VOTER_DEMOGRAPHICS_URLS.keys())]['table_1']['url']
VOTER_DEMOGRAPHICS_FILENAME = "wa_voter_demographics_tables.xlsx"
```

## Error Handling

The scraper includes comprehensive error handling:
- Network timeouts and connection errors
- Invalid URLs and file access issues
- Date parsing errors
- File download failures

## Rate Limiting

The scraper includes built-in rate limiting:
- 1-second delay between year searches
- Respectful user agent headers
- Timeout settings for requests

## Troubleshooting

### Common Issues

1. **No tables found**: The website structure may have changed. Check the search URLs in the scraper.

2. **Download failures**: 
   - Check network connectivity
   - Verify file URLs are still valid
   - Ensure sufficient disk space

3. **Constants file update fails**:
   - Check file permissions
   - Verify the constants file path
   - Ensure the file contains the expected VOTER_DEMOGRAPHICS_URL pattern

### Debug Mode

Enable debug logging by modifying the logging level in `voter_demographics_scraper.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new search capabilities:
1. Update the search terms in `VoterDemographicsScraper`
2. Add new date patterns if needed
3. Test with different year ranges
4. Update this README with new features

## License

This code is part of the ElectionModelling project and follows the same license terms. 