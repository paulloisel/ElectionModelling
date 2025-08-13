"""Example of using the ACS Feature Reduction Pipeline across multiple states."""

from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
import pandas as pd
import numpy as np

def main():
    # Initialize pipeline for the most recent years
    pipeline = ACSFeatureReductionPipeline(
        years=[2021, 2022],  # Get variables available in recent years
        dataset="acs/acs5"
    )
    
    # Load metadata for variables available across years
    metadata = pipeline.load_metadata()
    print(f"\nLoaded {len(metadata)} variables common across years 2021-2022")
    
    # Select demographic and socioeconomic variables
    selected = pipeline.select_variables(
        keywords=[
            "income", "poverty", "education", "employment",
            "age", "race", "ethnicity", "housing"
        ],
        table_prefixes=[
            "B19",  # Income
            "B17",  # Poverty
            "B15",  # Education
            "B23",  # Employment
            "B01",  # Age
            "B02",  # Race
            "B03",  # Ethnicity
            "B25"   # Housing
        ]
    )
    print(f"\nSelected {len(selected)} demographic and socioeconomic variables")
    
    # Simulate data for multiple states
    # In practice, this would be real census data
    states = ["CA", "WA", "OR"]
    n_counties = 20  # Simulate 20 counties per state
    
    all_data = []
    for state in states:
        # Create example data for each state
        state_data = pd.DataFrame({
            # Income variables
            "B19013_001E": np.random.normal(65000, 25000, n_counties),  # Median household income
            "B19013_002E": np.random.normal(63000, 24000, n_counties),  # Correlated with median income
            
            # Education variables
            "B15003_001E": np.random.normal(35, 8, n_counties),  # Education level
            "B15003_002E": np.random.normal(33, 7, n_counties),  # Correlated education metric
            
            # Employment variables
            "B23025_001E": np.random.normal(50000, 15000, n_counties),  # Employment
            "B23025_002E": np.random.normal(48000, 14000, n_counties),  # Correlated employment metric
        })
        
        # Add state identifier
        state_data["state"] = state
        all_data.append(state_data)
    
    # Combine all state data
    combined_data = pd.concat(all_data, ignore_index=True)
    print("\nCombined data shape:", combined_data.shape)
    
    # Remove highly correlated variables
    reduced = pipeline.reduce_dataframe(
        combined_data.drop("state", axis=1),  # Exclude state column from correlation check
        corr_threshold=0.8
    )
    
    # Add state back
    reduced["state"] = combined_data["state"]
    
    print(f"\nReduced from {len(combined_data.columns)-1} to {len(reduced.columns)-1} variables after correlation check")
    print("Remaining variables:", [col for col in reduced.columns if col != "state"])
    
    # Show summary by state
    print("\nSummary statistics by state:")
    print(reduced.groupby("state").mean().round(2))

if __name__ == "__main__":
    main() 