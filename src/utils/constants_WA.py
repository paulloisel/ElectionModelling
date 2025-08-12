"""
Constants for Washington State election data downloader.
"""

# Supported years for data download
SUPPORTED_YEARS = [2012, 2014, 2016, 2018, 2020, 2022, 2024]

# Election dates by year
# Format: {year: {'primary': (month, day), 'general': (month, day)}}
ELECTION_DATES = {
    2012: {'primary': (8, 7), 'general': (11, 6)},
    2014: {'primary': (8, 5), 'general': (11, 4)},
    2016: {'primary': (8, 2), 'general': (11, 8)},
    2018: {'primary': (8, 7), 'general': (11, 6)},
    2020: {'primary': (8, 4), 'general': (11, 3)},
    2022: {'primary': (8, 2), 'general': (11, 8)},
    2024: {'primary': (8, 6), 'general': (11, 5)},
}

# Special URL patterns for registered voters data
# Some years have different URL patterns or are in ZIP files
REGISTERED_VOTERS_URLS = {
    2012: {
        'general': {
            'url': 'https://www2.sos.wa.gov/_assets/elections/research/2012-general-data2.zip',
            'filename': '2012-general-data/FINAL 2012Gen - Cong.xlsx',
            'is_zip': True,
            'note': 'Contains both results and registration data in same file'
        }
    },
    2014: {
        'primary': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2022-05/2014-primary-data.zip',
            'filename': '2014-primary-data/AbstractResults_Leg_2014Primary.txt',
            'is_zip': True
        },
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2023-05/2014-general-data.zip',
            'filename': '2014-general-data/Congressional, Legislative District Breakdowns/FINAL 2014Gen - Cong.xlsx',
            'is_zip': True
        }
    },
    2016: {
        'primary': {
            'url': 'https://www.sos.wa.gov/_assets/elections/research/2016-primary-data.zip',
            'filename': '2016-primary-data/Congressional Legislative District Breakdowns/FINAL 2016Pri - Cong.xlsx',
            'is_zip': True
        },
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2023-05/2016-general-data.zip',
            'filename': '2016-general-data/Congressional and Legislative District Breakdowns/FINAL 2016Gen - Cong.xlsx',
            'is_zip': True
        }
    },
    2018: {
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2022-05/2018-general-data.zip',
            'filename': '2018-general-data/Congressional and Legislative District Breakdowns (Statewide Contests Only)/2018Gen - Cong.xlsx',
            'is_zip': True
        }
    },
    2020: {
        'primary': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2022-05/Congressional%2520and%2520Legislative%2520District%2520Breakdowns.zip',
            'filename': 'Results by Congressional District - 2020 Primary.xlsx',
            'is_zip': True
        },
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2022-05/2020Gen%2520Results%2520by%2520Congressional%2520District.xlsx',
            'filename': None,
            'is_zip': False
        }
    },
    2022: {
        'primary': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2023-05/2022%20primary%20results%20by%20congressional%20district.xlsx',
            'filename': None,
            'is_zip': False
        },
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2023-05/2022gen%20results%20by%20congressional%20district.xlsx',
            'filename': None,
            'is_zip': False
        }
    },
    2024: {
        'general': {
            'url': 'https://www.sos.wa.gov/sites/default/files/2024-12/2024Gen%20Results%20by%20Congressional%20District.xlsx',
            'filename': None,
            'is_zip': False
        }
    }
}

# Base URLs for data sources
SOS_BASE_URL = "https://www.sos.wa.gov/sites/default/files"
RESULTS_BASE_URL = "https://results.vote.wa.gov/results"

# File naming patterns
SOS_FILENAME_PATTERN = "{year}{election_type}%20Results%20by%20Congressional%20District.xlsx"
RESULTS_FILENAME_PATTERN = "{date_str}_Congressional.csv"

# Local file naming patterns
LOCAL_SOS_FILENAME_PATTERN = "wa_{year}_{election_type}_registered_voters.xlsx"
LOCAL_RESULTS_FILENAME_PATTERN = "wa_{date_str}_election_results.csv"

# Election types
ELECTION_TYPES = {
    'primary': 'Pri',
    'general': 'Gen'
}

# Special URL patterns for election results data
# Some years have different URL patterns with timestamps
ELECTION_RESULTS_URLS = {
    2012: {
        'primary': {
            'url': 'https://results.vote.wa.gov/results/20120807/export/20120807_congressional_20120904_1126.csv',
            'local_filename': 'wa_20120807_election_results.csv'
        },
        'general': {
            'url': 'https://results.vote.wa.gov/results/20121106/export/20121106_congressional_20121205_1451.csv',
            'local_filename': 'wa_20121106_election_results.csv'
        }
    }
}

# File validation settings
MIN_FILE_SIZE_BYTES = 1000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3

# User agent for requests
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Geographic data constants
GEOGRAPHIC_DATA_2012 = {
    'wa_congressional_districts_url': 'https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/53_WASHINGTON/53/tl_2020_53_cd113.zip',
    'wa_congressional_districts_filename': 'wa_congressional_districts_2012_redistricting_113th_115th_congress.zip',
    'description': 'Post-2010 redistricting boundaries (113th-115th Congresses)'
}

# Census data constants
CENSUS_YEARS = range(2010, 2024)  # 2010-2023 5-year releases
CENSUS_VARIABLES = ["B01001_001E"]  # Total population - add more variables as needed
CENSUS_BASE_URL = "https://api.census.gov/data/{yr}/acs/acs5"

# Washington State specific census constants
WA_STATE_FIPS = "53"  # Washington State FIPS code
WA_CENSUS_GEO = [("state", WA_STATE_FIPS), ("congressional district", "*")]

# Voter demographics data
VOTER_DEMOGRAPHICS_URL = "https://www.sos.wa.gov/sites/default/files/2023-06/Voter%20Demographics%20Tables.xlsx"
VOTER_DEMOGRAPHICS_FILENAME = "wa_voter_demographics_tables.xlsx" 
