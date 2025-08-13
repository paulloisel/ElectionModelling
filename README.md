# Election Modeling

A comprehensive election turnout modeling project that analyzes historical election data from Washington states, incorporating demographic data from the American Community Survey (ACS) to predict voter turnout.

## Project Structure

```
ElectionModelling/
├── data/
│   ├── raw/                # PDFs, XLSX, shapefiles, ACS zips
│   └── processed/          # cleaned Parquet/Feather
├── notebooks/
│   └── 01_turnout_model_walkthrough.ipynb  <- final explainer
├── src/
│   ├── ingest/
│   │   ├── wa_results_loader.py    # XLSX & CSV → parquet
│   │   └── census_fetch.py         # ACS pull & tidy
│   ├── features/
│   │   └── build_features.py       # joins, lag features, demographics
│   ├── models/
│   │   ├── train_elasticnet.py
│   │   ├── train_xgboost.py
│   │   └── train_mixedlm.py        # random-intercept by state
│   └── utils/geometry.py           # spatial joins for CA reg counts
├── environment.yml                 # pin pandas, xgboost, statsmodels, pdfplumber
└── README.md
```

## Overview

This project implements multiple modeling approaches to predict election turnout:

- **ElasticNet Regression**: Linear model with L1/L2 regularization
- **XGBoost**: Gradient boosting for non-linear relationships
- **Mixed Linear Models**: Random intercepts by state for hierarchical data

## Data Sources

### Election Data:
- **Washington State Elections**: XLSX and CSV files from state election offices

### Demographic Data
- **American Community Survey (ACS)**: Census Bureau demographic variables
- **Washington State Voter Demographics**: Age-based voter demographics from Secretary of State (only 2024 for now)
- **Spatial Data**: Shapefiles for precinct boundaries and spatial joins

## Features

### Data Ingestion (`src/ingest/`)
- **WA Results Loader**: Processes XLSX and CSV election files
- **Census Fetcher**: Pulls ACS data via Census API and tidies for analysis

### Feature Engineering (`src/features/`)
- **Data Joins**: Combines election results with demographic data
- **Lag Features**: Creates time-series features for historical turnout patterns
- **Demographic Features**: Engineers variables from raw census data
- **Interaction Features**: Creates meaningful variable interactions

### Modeling (`src/models/`)
- **ElasticNet**: Linear model with regularization for feature selection
- **XGBoost**: Gradient boosting with feature importance analysis
- **Mixed Linear Models**: Hierarchical models with state-level random effects

### Utilities (`src/utils/`)
- **Geometry**: Spatial operations for precinct-level analysis
- **Downloader**: Automated data downloader for Washington State election data and voter demographics

## Setup

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ElectionModelling
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate election-modeling
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, xgboost, statsmodels, pdfplumber; print('All dependencies installed!')"
   ```

### Data Preparation

1. **Place raw data in `data/raw/`**:
   - WA election XLSX/CSV files
   - Census shapefiles
   - ACS data zips

2. **Run data ingestion scripts**:
   ```bash
   python src/ingest/ca_sov_parser.py
   python src/ingest/wa_results_loader.py
   python src/ingest/census_fetch.py
   ```

## Usage

### Quick Start

1. **Open the walkthrough notebook**:
   ```bash
   jupyter notebook notebooks/01_turnout_model_walkthrough.ipynb
   ```

2. **Follow the notebook sections**:
   - Data Ingestion
   - Feature Engineering
   - Model Training
   - Evaluation

### Individual Components

**Download Washington State Data**:
```python
from src.utils.downloader import WAStateDownloader

# Initialize downloader
downloader = WAStateDownloader(output_dir="data/raw/wa")

# Download voter demographics data
demographics_path = downloader.download_voter_demographics()

# Download election data for specific year
results = downloader.download_specific_year_data(2024)

# Download all available data
all_results = downloader.download_all_even_years_data()
```

**Train ElasticNet Model**:
```python
from src.models.train_elasticnet import train_elasticnet, evaluate_model
# Train and evaluate model
```

**Train XGBoost Model**:
```python
from src.models.train_xgboost import train_xgboost, feature_importance_analysis
# Train model and analyze feature importance
```

**Train Mixed Linear Model**:
```python
from src.models.train_mixedlm import train_mixedlm
# Train hierarchical model with state random effects
```

## Model Performance

The project compares three modeling approaches:

1. **ElasticNet**: Good for feature selection and interpretability
2. **XGBoost**: Best for capturing non-linear relationships
3. **Mixed Linear Models**: Accounts for state-level heterogeneity

## Data Notes

### Washington State Election Data

The Washington state election data has several important characteristics that should be noted:

#### Data Structure Variations
- **2012 General Election**: Results and registration data are combined in the same file (`FINAL 2012Gen - Cong.xlsx`)
- **2014 Primary**: Data is stored as a text file (`AbstractResults_Leg_2014Primary.txt`) rather than Excel format
- **ZIP Archives**: Many years store data in ZIP files with nested directory structures
- **File Naming**: Inconsistent naming conventions across years (e.g., "Congressional, Legislative District Breakdowns" vs "Congressional Legislative District Breakdowns")

#### Data Availability
- **2012**: Only general election data available (no primary)
- **2014-2018**: Limited registered voters data available
- **2020-2024**: Complete data available for both primary and general elections
- **Election Results**: Available for all years 2014-2024 (except 2012)

#### File Formats
- **Excel Files**: Most recent years (2020-2024) use standard Excel format
- **Text Files**: 2014 primary data uses text format
- **ZIP Archives**: Historical data often packaged in ZIP files with complex directory structures
- **CSV Files**: Election results consistently available in CSV format from results.vote.wa.gov

#### Data Quality Considerations
- **Inconsistent Column Structures**: Different years may have different column names and structures
- **Missing Data**: Some historical years have incomplete data availability
- **Format Changes**: Data format and structure has evolved over time
- **Spatial Aggregation**: Data is aggregated at congressional district level, not precinct level

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request