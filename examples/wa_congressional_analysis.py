"""Analysis of Washington congressional districts census data from 2012-2020."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
from src.ingest.censuspipeline.metadata import fetch_variable_metadata
from src.utils.downloader import CensusDownloader
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wa_congressional_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def get_wa_congressional_data(year, variables, dataset="acs/acs5"):
    """Fetch real census data for WA congressional districts for a specific year."""
    try:
        # Use the CensusDownloader to fetch real census data
        census_downloader = CensusDownloader(output_dir="data/raw/census")
        
        # Download real census data for the specified year and variables
        df = census_downloader.download_wa_congressional_districts(
            year=year, 
            variables=variables
        )
        
        if df is not None:
            logging.info(f"Successfully fetched real census data for year {year}")
            logging.info(f"Data shape: {df.shape}")
            logging.info(f"Variables: {list(df.columns)}")
            return df
        else:
            logging.warning(f"Failed to fetch real census data for year {year}, falling back to simulated data")
            return _generate_simulated_data(year, variables)
            
    except Exception as e:
        logging.error(f"Error fetching real census data for year {year}: {str(e)}")
        logging.info("Falling back to simulated data for demonstration purposes")
        return _generate_simulated_data(year, variables)

def _generate_simulated_data(year, variables):
    """Generate simulated data as fallback when real census data is unavailable."""
    n_districts = 10  # WA congressional districts
    
    data = {}
    for var in variables:
        # Generate data based on variable type
        if 'MEDIAN' in var or 'INCOME' in var.upper():
            base = 60000
            std = 20000
        elif 'TOTAL' in var.upper():
            base = 50000
            std = 15000
        else:
            base = 1000
            std = 300
        
        # Add slight year-over-year trend
        year_factor = (year - 2012) * 0.02  # 2% increase per year
        base = base * (1 + year_factor)
        
        data[var] = np.random.normal(base, std, n_districts)
    
    df = pd.DataFrame(data)
    df['district'] = [f"WA-{i+1}" for i in range(n_districts)]
    df['year'] = year
    return df

def select_variables_hardcoded(metadata, max_variables=20):
    """Select variables using hardcoded list of diverse demographic variables."""
    logging.info("Using HARDCODED variable selection approach")
    
    # Select specific diverse variables that won't be highly correlated
    # These are key demographic variables from different categories
    specific_variables = [
        'B01001_001E',  # Total population
        'B19013_001E',  # Median household income
        'B15003_022E',  # Bachelor's degree
        'B15003_023E',  # Master's degree
        'B25003_002E',  # Renter occupied housing
        'B25003_001E',  # Total occupied housing
        'B23025_002E',  # In labor force
        'B23025_001E',  # Total population 16+
        'B17001_002E',  # Income below poverty level
        'B17001_001E',  # Total population for poverty determination
        'B01001_002E',  # Male population
        'B01001_026E',  # Female population
        'B25064_001E',  # Median gross rent
        'B25077_001E',  # Median home value
        'B08303_001E',  # Total commuters
        'B08303_010E',  # 60+ minute commute
        'B25002_002E',  # Occupied housing units
        'B25002_003E',  # Vacant housing units
        'B25035_001E',  # Median year structure built
        'B25040_001E'   # Total housing units by heating fuel
    ]
    
    # Filter to only include variables that exist in our metadata
    available_variables = [var for var in specific_variables if var in metadata['name'].values]
    variables = available_variables[:max_variables]  # Take first N available
    
    logging.info(f"Selected {len(variables)} hardcoded demographic variables")
    logging.info(f"Variables: {variables}")
    
    return variables

def select_variables_automated(pipeline, metadata, max_variables=20):
    """Select variables using automated feature reduction pipeline."""
    logging.info("Using AUTOMATED variable selection approach")
    
    # Select variables by categories - use more diverse prefixes to avoid high correlations
    categories = {
        'Demographics': ['B01001'],  # Total population
        'Income': ['B19013'],        # Median income
        'Education': ['B15003'],     # Educational attainment
        'Housing': ['B25003'],       # Tenure
        'Employment': ['B23025'],    # Employment status
        'Poverty': ['B17001']        # Poverty status
    }
    
    # Get variables for each category
    selected_vars = []
    for category, prefixes in categories.items():
        vars_in_category = pipeline.select_variables(table_prefixes=prefixes)
        if not vars_in_category.empty:
            selected_vars.append(vars_in_category)
            logging.info(f"{category}: {len(vars_in_category)} variables")
    
    all_selected = pd.concat(selected_vars, ignore_index=True)
    variables = list(all_selected['name'].unique())
    
    # Limit to a reasonable number of variables
    if len(variables) > max_variables:
        logging.info(f"Limiting variables from {len(variables)} to {max_variables} for faster processing")
        variables = variables[:max_variables]
    
    logging.info(f"Selected {len(variables)} automated demographic variables")
    logging.info(f"Variables: {variables}")
    
    return variables

def main(use_hardcoded_selection=True, max_variables=20, years=None, corr_threshold=0.95):
    """
    Main analysis function with configurable variable selection approach.
    
    Parameters:
    -----------
    use_hardcoded_selection : bool, default True
        If True, use hardcoded list of diverse variables
        If False, use automated feature reduction pipeline
    max_variables : int, default 20
        Maximum number of variables to select
    years : list, optional
        Years to analyze (default: 2018-2020)
    corr_threshold : float, default 0.95
        Correlation threshold for feature reduction
    """
    # Initialize pipeline
    if years is None:
        years = list(range(2018, 2021))  # 2018 to 2020
    
    pipeline = ACSFeatureReductionPipeline(
        years=years,
        output_dir="data/processed/test_examples"
    )
    
    logging.info("Starting WA congressional district analysis")
    logging.info(f"Analyzing years: {years}")
    logging.info(f"Variable selection approach: {'HARDCODED' if use_hardcoded_selection else 'AUTOMATED'}")
    logging.info(f"Max variables: {max_variables}")
    logging.info(f"Correlation threshold: {corr_threshold}")
    
    # Load metadata for variables available across all years
    logging.info("Loading common variables across years...")
    metadata = pipeline.load_metadata()
    logging.info(f"Found {len(metadata)} variables common across all years")
    
    # Select variables based on approach
    if use_hardcoded_selection:
        variables = select_variables_hardcoded(metadata, max_variables)
        # Set the selected variables in the pipeline for later use
        selected_metadata = metadata[metadata['name'].isin(variables)]
        pipeline._selected = selected_metadata
    else:
        variables = select_variables_automated(pipeline, metadata, max_variables)
    
    # Collect data for all years
    all_data = []
    real_data_count = 0
    simulated_data_count = 0
    
    for year in years:
        logging.info(f"Processing year {year}...")
        year_data = get_wa_congressional_data(year, variables)
        if year_data is not None:
            all_data.append(year_data)
            # Check if this was real or simulated data by looking for the 'NAME' column
            # Real census data has 'NAME' column, simulated data has 'district' column
            if 'NAME' in year_data.columns:
                real_data_count += 1
                logging.info(f"Year {year}: Using REAL census data")
            else:
                simulated_data_count += 1
                logging.info(f"Year {year}: Using SIMULATED data")
    
    logging.info(f"\nData Summary:")
    logging.info(f"  Years with real census data: {real_data_count}")
    logging.info(f"  Years with simulated data: {simulated_data_count}")
    logging.info(f"  Total years processed: {len(all_data)}")
    
    # Combine all years
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined data shape: {combined_data.shape}")
    
    # Handle both real and simulated data formats
    # Real census data uses 'NAME' column, simulated data uses 'district' column
    if 'NAME' in combined_data.columns:
        district_col = 'NAME'
        logging.info("Processing real census data format")
    else:
        district_col = 'district'
        logging.info("Processing simulated data format")
    
    # Remove highly correlated variables for each year
    results_by_year = {}
    for year in years:
        year_data = combined_data[combined_data['year'] == year].copy()
        data_for_correlation = year_data.drop([district_col, 'year'], axis=1)
        
        reduced = pipeline.reduce_dataframe(
            data_for_correlation,
            corr_threshold=corr_threshold
        )
        reduced[district_col] = year_data[district_col]
        reduced['year'] = year
        
        results_by_year[year] = reduced
        logging.info(f"Year {year}: Reduced to {len(reduced.columns)-2} variables")
    
    # Combine results and save
    final_results = pd.concat(list(results_by_year.values()), ignore_index=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    approach_suffix = "hardcoded" if use_hardcoded_selection else "automated"
    pipeline.save_results(
        reduced_df=final_results,
        metadata_df=metadata[metadata['name'].isin(final_results.columns)],
        data_filename=f"wa_congressional_analysis_{approach_suffix}_{timestamp}.csv",
        metadata_filename=f"wa_congressional_variables_{approach_suffix}_{timestamp}.csv"
    )
    
    # Generate summary statistics
    logging.info("\nSummary of changes over time:")
    for year in years:
        year_data = final_results[final_results['year'] == year]
        logging.info(f"\nYear {year} summary:")
        # Convert censusgeo objects to strings for sorting
        districts = [str(d) for d in year_data[district_col].unique()]
        for district in sorted(districts):
            # Find the matching district data by converting to string for comparison
            district_data = year_data[year_data[district_col].astype(str) == district]
            logging.info(f"{district}:")
            for col in district_data.columns:
                if col not in [district_col, 'year']:
                    val = district_data[col].iloc[0]
                    logging.info(f"  {col}: {val:.2f}")

if __name__ == "__main__":
    # Example usage with different approaches
    print("Running analysis with HARDCODED variable selection...")
    main(use_hardcoded_selection=True, max_variables=20)
    
    print("\n" + "="*50)
    print("Running analysis with AUTOMATED variable selection...")
    main(use_hardcoded_selection=False, max_variables=20)

