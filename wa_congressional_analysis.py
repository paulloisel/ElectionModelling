"""Analysis of Washington congressional districts census data from 2012-2020."""

import pandas as pd
import numpy as np
from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
from src.ingest.censuspipeline.metadata import fetch_variable_metadata
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
    """Fetch data for WA congressional districts for a specific year."""
    try:
        # We'll simulate the data for now - in practice this would fetch from Census API
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
    except Exception as e:
        logging.error(f"Error fetching data for year {year}: {str(e)}")
        return None

def main():
    # Initialize pipeline
    years = list(range(2012, 2021))  # 2012 to 2020
    pipeline = ACSFeatureReductionPipeline(years=years)
    
    logging.info("Starting WA congressional district analysis")
    logging.info(f"Analyzing years: {years}")
    
    # Load metadata for variables available across all years
    logging.info("Loading common variables across years...")
    metadata = pipeline.load_metadata()
    logging.info(f"Found {len(metadata)} variables common across all years")
    
    # Select variables by categories
    categories = {
        'Demographics': ['B01', 'B02', 'B03'],
        'Social': ['B07', 'B08', 'B09', 'B10', 'B11'],
        'Economic': ['B19', 'B20', 'B21', 'B22', 'B23'],
        'Housing': ['B25', 'B26'],
        'Education': ['B14', 'B15'],
        'Family': ['B11', 'B12', 'B13'],
        'Language': ['B16'],
        'Poverty': ['B17'],
        'Healthcare': ['B27', 'B28'],
        'Veterans': ['B21']
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
    logging.info(f"Total selected variables: {len(variables)}")
    
    # Collect data for all years
    all_data = []
    for year in years:
        logging.info(f"Processing year {year}...")
        year_data = get_wa_congressional_data(year, variables)
        if year_data is not None:
            all_data.append(year_data)
    
    # Combine all years
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info(f"Combined data shape: {combined_data.shape}")
    
    # Remove highly correlated variables for each year
    results_by_year = {}
    for year in years:
        year_data = combined_data[combined_data['year'] == year].copy()
        data_for_correlation = year_data.drop(['district', 'year'], axis=1)
        
        reduced = pipeline.reduce_dataframe(
            data_for_correlation,
            corr_threshold=0.8
        )
        reduced['district'] = year_data['district']
        reduced['year'] = year
        
        results_by_year[year] = reduced
        logging.info(f"Year {year}: Reduced to {len(reduced.columns)-2} variables")
    
    # Combine results and save
    final_results = pd.concat(list(results_by_year.values()), ignore_index=True)
    
    # Save results
    output_file = f'wa_congressional_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    final_results.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    
    # Save metadata for selected variables
    metadata_subset = metadata[metadata['name'].isin(final_results.columns)]
    metadata_file = f'wa_congressional_variables_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metadata_subset.to_csv(metadata_file, index=False)
    logging.info(f"Variable metadata saved to {metadata_file}")
    
    # Generate summary statistics
    logging.info("\nSummary of changes over time:")
    for year in years:
        year_data = final_results[final_results['year'] == year]
        logging.info(f"\nYear {year} summary:")
        for district in sorted(year_data['district'].unique()):
            district_data = year_data[year_data['district'] == district]
            logging.info(f"{district}:")
            for col in district_data.columns:
                if col not in ['district', 'year']:
                    val = district_data[col].iloc[0]
                    logging.info(f"  {col}: {val:.2f}")

if __name__ == "__main__":
    main()

