# Examples

This directory contains example scripts that demonstrate how to use the Election Modeling project's census pipeline and analysis tools.

## Pipeline Architecture

The census analysis pipeline follows this workflow:

```
1. Data Source Selection
   ├── Real Census API Data (primary)
   └── Simulated Data (fallback)

2. Variable Selection (Two Approaches)
   ├── Hardcoded Selection (fast, predictable)
   └── Automated Selection (comprehensive, production-ready)

3. Feature Reduction
   ├── Correlation Analysis
   ├── Variable Filtering
   └── Final Dataset Creation

4. Output Generation
   ├── Processed Data CSV
   ├── Variable Metadata CSV
   └── Analysis Logs
```

## Variable Selection Approaches

### 1. Hardcoded Selection (`use_hardcoded_selection=True`)
- **Purpose**: Fast, predictable variable selection for demonstrations and testing
- **Method**: Uses a carefully curated list of diverse demographic variables
- **Advantages**:
  - Fast execution
  - Predictable results
  - Avoids highly correlated variables
  - Good for demonstrations
- **Variables Selected**:
  - Total population, income, education, housing, employment, poverty
  - 20 diverse demographic indicators

### 2. Automated Selection (`use_hardcoded_selection=False`)
- **Purpose**: Comprehensive variable discovery using the full feature reduction pipeline
- **Method**: Uses category-based filtering and automated selection
- **Advantages**:
  - More comprehensive variable discovery
  - Can handle larger variable sets
  - Better for production use
  - Leverages full pipeline capabilities
- **Process**:
  1. Load metadata for all available variables
  2. Filter by demographic categories (income, education, housing, etc.)
  3. Apply automated selection criteria
  4. Limit to specified number of variables

## Feature Reduction Pipeline

Both approaches use the same feature reduction pipeline:

### Step 1: Data Fetching
```python
# Fetch real census data from Census API
census_downloader = CensusDownloader()
df = census_downloader.download_wa_congressional_districts(year, variables)
```

### Step 2: Variable Selection
```python
# Approach 1: Hardcoded
variables = select_variables_hardcoded(metadata, max_variables=20)

# Approach 2: Automated
variables = select_variables_automated(pipeline, metadata, max_variables=20)
```

### Step 3: Correlation Analysis
```python
# Remove highly correlated variables
reduced = pipeline.reduce_dataframe(data, corr_threshold=0.95)
```

### Step 4: Output Generation
```python
# Save processed results
pipeline.save_results(reduced_df, metadata_df, filename)
```

## Example Scripts

### Census Pipeline Examples

- **`census_example.py`** - Basic example of using the ACS Feature Reduction Pipeline
  - Shows how to load metadata, filter variables by keywords, and reduce correlations
  - Demonstrates OpenAI-powered variable selection (requires API key)
  - Saves results to `data/processed/test_examples/`

- **`multi_state_census_example.py`** - Multi-state analysis example
  - Demonstrates processing census data across multiple states (CA, WA, OR)
  - Shows how to handle state-level data aggregation
  - Includes correlation-based feature reduction

### Washington State Analysis Examples

- **`wa_congressional_analysis.py`** - Complete Washington congressional district analysis
  - **Uses real Census API data when available** (falls back to simulated data if needed)
  - **Supports both hardcoded and automated variable selection**
  - Analyzes census data for WA congressional districts from 2012-2020
  - Demonstrates year-over-year trend analysis
  - Includes comprehensive logging and data processing
  - Saves timestamped results with approach-specific filenames

- **`visualize_wa_congressional.py`** - Visualization of WA congressional data
  - Creates plots and charts from congressional district analysis
  - Generates PNG files for data visualization

### Demographics Search Examples

- **`example_demographics_search.py`** - Demographics table search example
  - Shows how to search for voter demographics tables
  - Demonstrates updating constants and metadata
  - Includes comprehensive error handling

### Testing Examples

- **`test_real_census_data.py`** - Test real census data fetching
  - Demonstrates fetching actual census data from the Census API
  - Shows real demographic values for Washington congressional districts
  - Tests both single year and multi-year data downloads

- **`test_variable_selection_approaches.py`** - Test both variable selection approaches
  - Demonstrates hardcoded vs automated variable selection
  - Shows advantages and use cases for each approach
  - Compares results and performance

## How to Run Examples

All examples can be run from the project root directory:

```bash
# Run from project root
python3 examples/census_example.py
python3 examples/multi_state_census_example.py
python3 examples/wa_congressional_analysis.py
python3 examples/test_real_census_data.py
python3 examples/test_variable_selection_approaches.py
```

### Running with Different Approaches

```python
# Hardcoded selection (fast, predictable)
main(use_hardcoded_selection=True, max_variables=20)

# Automated selection (comprehensive, production-ready)
main(use_hardcoded_selection=False, max_variables=20)

# Custom parameters
main(
    use_hardcoded_selection=True,
    max_variables=15,
    years=[2019, 2020],
    corr_threshold=0.90
)
```

## Output

Most examples save their results to `data/processed/test_examples/` with descriptive filenames:
- `*_reduced_data.csv` - Processed and reduced census data
- `*_variables.csv` - Metadata for selected variables
- `*_analysis_hardcoded_*.csv` - Results using hardcoded selection
- `*_analysis_automated_*.csv` - Results using automated selection

## Requirements

- Python 3.7+
- Required packages: pandas, numpy, requests
- Optional: OpenAI API key for AI-powered variable selection
- Census API access for real data (examples use simulated data as fallback)

## Notes

- **Updated examples now use real Census API data when available**
- Examples fall back to simulated data if Census API is unavailable or fails
- **Two variable selection approaches available**:
  - Hardcoded: Fast, predictable, good for demos
  - Automated: Comprehensive, production-ready
- Both approaches use the same feature reduction pipeline
- Some examples require API keys (OpenAI, Census) for full functionality
- All examples include error handling and logging

## Pipeline Configuration

### Key Parameters

- `use_hardcoded_selection`: Choose variable selection approach
- `max_variables`: Limit number of variables for processing
- `years`: Specify years to analyze
- `corr_threshold`: Correlation threshold for feature reduction (0.0-1.0)

### Performance Considerations

- **Hardcoded selection**: Faster, uses 20 pre-selected diverse variables
- **Automated selection**: Slower, can process hundreds of variables
- **Real data**: Requires internet connection and Census API access
- **Simulated data**: Works offline, good for testing pipeline logic
