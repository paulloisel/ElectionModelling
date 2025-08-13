"""Test census pipeline for Washington state congressional districts (2012-2022)."""

import pandas as pd
import numpy as np
from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
import random

def main():
    # Initialize pipeline for 2012-2022
    pipeline = ACSFeatureReductionPipeline(
        years=list(range(2012, 2023)),  # 2012 to 2022
        dataset="acs/acs5"
    )
    
    print("\nLoading variables common across 2012-2022...")
    metadata = pipeline.load_metadata()
    print(f"Found {len(metadata)} common variables")
    
    # Define diverse category prefixes for different types of data
    table_categories = {
        'Demographics': ['B01', 'B02', 'B03'],  # Age, Race, Hispanic Origin
        'Social': ['B07', 'B08', 'B09', 'B10', 'B11'],  # Migration, Journey to Work, Disability
        'Economic': ['B19', 'B20', 'B21', 'B22', 'B23'],  # Income, Earnings, Employment
        'Housing': ['B25', 'B26'],  # Housing Characteristics
        'Education': ['B14', 'B15'],  # School Enrollment, Educational Attainment
        'Family': ['B11', 'B12', 'B13'],  # Household, Family Type
        'Language': ['B16'],  # Language Spoken at Home
        'Poverty': ['B17'],  # Poverty Status
        'Healthcare': ['B27', 'B28'],  # Health Insurance
        'Veterans': ['B21']  # Veteran Status
    }
    
    # Select variables from each category
    selected_vars = []
    print("\nSelecting variables from each category:")
    for category, prefixes in table_categories.items():
        # Filter variables for this category
        category_vars = pipeline.select_variables(table_prefixes=prefixes)
        if not category_vars.empty:
            # Randomly select variables from this category
            n_vars = max(5, len(category_vars) // 100)  # Take at least 5 vars or 1% of available vars
            selected = category_vars.sample(n=min(n_vars, len(category_vars)), random_state=42)
            selected_vars.append(selected)
            print(f"{category}: {len(selected)} variables")
    
    # Combine all selected variables
    final_metadata = pd.concat(selected_vars, ignore_index=True)
    print(f"\nTotal selected variables: {len(final_metadata)}")
    
    # Print example variables from each category
    print("\nExample variables selected:")
    for category, prefixes in table_categories.items():
        category_vars = final_metadata[final_metadata['name'].str.startswith(tuple(prefixes))]
        if not category_vars.empty:
            print(f"\n{category}:")
            print(category_vars[['name', 'label']].head(2))
    
    # Simulate data for WA congressional districts
    n_districts = 10  # WA has 10 congressional districts
    print(f"\nSimulating data for {n_districts} WA congressional districts...")
    
    # Create simulated data
    sim_data = {}
    for var in final_metadata['name']:
        # Generate random data with reasonable ranges based on variable type
        if 'MEDIAN' in var or 'INCOME' in var.upper():
            sim_data[var] = np.random.normal(60000, 20000, n_districts)
        elif 'TOTAL' in var.upper():
            sim_data[var] = np.random.normal(50000, 15000, n_districts)
        else:
            sim_data[var] = np.random.normal(1000, 300, n_districts)
    
    df = pd.DataFrame(sim_data)
    df['district'] = [f"WA-{i+1}" for i in range(n_districts)]
    
    # Remove highly correlated variables
    print("\nRemoving highly correlated variables...")
    reduced = pipeline.reduce_dataframe(
        df.drop('district', axis=1),
        corr_threshold=0.8
    )
    reduced['district'] = df['district']
    
    print(f"\nReduced from {len(df.columns)-1} to {len(reduced.columns)-1} variables")
    print("\nFinal variable count by prefix:")
    for category, prefixes in table_categories.items():
        vars_count = sum(1 for col in reduced.columns if any(col.startswith(prefix) for prefix in prefixes))
        if vars_count > 0:
            print(f"{category}: {vars_count}")
    
    # Save results
    print("\nSaving results...")
    reduced.to_csv('wa_congressional_test_data.csv', index=False)
    final_metadata.to_csv('wa_congressional_variables_metadata.csv', index=False)
    print("Results saved to wa_congressional_test_data.csv and wa_congressional_variables_metadata.csv")

if __name__ == "__main__":
    main()
