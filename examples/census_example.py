"""Example usage of the ACS Feature Reduction Pipeline."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
from src.ingest.censuspipeline.openai_selector import OpenAISelector

def main():
    # Initialize the pipeline for multiple years
    pipeline = ACSFeatureReductionPipeline(
        years=[2020, 2021, 2022],  # Get variables available across these years
        dataset="acs/acs5",
        output_dir="data/processed/test_examples"
    )
    
    # Load metadata (this will get variables common across all specified years)
    metadata = pipeline.load_metadata()
    print(f"\nLoaded {len(metadata)} variables common across years 2020-2022")
    
    # Example 1: Filter variables by keywords
    filtered = pipeline.select_variables(
        keywords=["income", "education", "employment"],
        table_prefixes=["B19", "B15", "B23"]  # Tables related to income, education, employment
    )
    print(f"\nFiltered to {len(filtered)} variables related to income, education, and employment")
    print("\nExample variables:")
    print(filtered[["name", "label"]].head())
    
    # Example 2: Use OpenAI to select most relevant variables
    # Note: Requires OpenAI API key to be set
    try:
        openai_selector = OpenAISelector()
        pipeline.openai_selector = openai_selector
        
        # Select top variables using OpenAI
        selected = pipeline.select_variables(
            keywords=["income", "education"],
            openai_top_k=5  # Get top 5 most relevant variables
        )
        print("\nTop 5 variables selected by OpenAI:")
        print(selected[["name", "label"]])
    except Exception as e:
        print("\nSkipped OpenAI selection (requires API key):", str(e))
    
    # Example 3: Remove highly correlated variables from actual data
    import pandas as pd
    import numpy as np
    
    # Create some example census data
    n_samples = 100
    example_data = pd.DataFrame({
        "B19013_001E": np.random.normal(60000, 20000, n_samples),  # Median income
        "B19013_002E": np.random.normal(58000, 19000, n_samples),  # Highly correlated with median income
        "B15003_001E": np.random.normal(40, 10, n_samples),        # Education years (independent)
    })
    
    # Reduce correlations
    reduced = pipeline.reduce_dataframe(example_data, corr_threshold=0.8)
    print(f"\nReduced from {len(example_data.columns)} to {len(reduced.columns)} variables after correlation check")
    print("Remaining variables:", list(reduced.columns))
    
    # Save results
    print("\nSaving results...")
    pipeline.save_results(
        reduced_df=reduced,
        metadata_df=filtered,
        data_filename="census_example_reduced_data.csv",
        metadata_filename="census_example_variables.csv"
    )

if __name__ == "__main__":
    main()