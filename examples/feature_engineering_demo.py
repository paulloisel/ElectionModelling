"""
Feature Engineering Demonstration

This script demonstrates the complete feature engineering pipeline
for election modeling using Washington State data.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.build_features import prepare_modeling_dataset


def load_multiple_years_data():
    """Load multiple years of election and demographics data."""
    
    print("Loading multiple years of data...")
    
    # Load multiple years of census data
    years = [2018, 2019, 2020]
    all_demographics = []
    
    for year in years:
        census_file = f'data/raw/census/wa_census_congressional_district_{year}.csv'
        if os.path.exists(census_file):
            df = pd.read_csv(census_file)
            all_demographics.append(df)
            print(f"  Loaded {year} census data: {df.shape}")
    
    if not all_demographics:
        print("No census data found")
        return None, None
    
    # Combine demographics data
    combined_demographics = pd.concat(all_demographics, ignore_index=True)
    print(f"  Combined demographics shape: {combined_demographics.shape}")
    
    # Load election data (we'll use 2024 as the target year)
    election_file = 'data/raw/wa_20241105_election_results.csv'
    if os.path.exists(election_file):
        election_data = pd.read_csv(election_file)
        print(f"  Loaded 2024 election data: {election_data.shape}")
    else:
        print("Election data not found")
        return None, None
    
    return election_data, combined_demographics


def demonstrate_feature_engineering():
    """Demonstrate the complete feature engineering pipeline."""
    
    print("=" * 80)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    election_data, demographics_data = load_multiple_years_data()
    
    if election_data is None or demographics_data is None:
        print("❌ Failed to load data")
        return
    
    print("\n" + "=" * 50)
    print("PREPARING MODELING DATASET")
    print("=" * 50)
    
    # Prepare modeling dataset with different target variables
    target_variables = ['democratic_share', 'total_votes', 'margin']
    
    for target_var in target_variables:
        print(f"\n--- Preparing dataset for target: {target_var} ---")
        
        # Prepare dataset
        final_df, feature_cols = prepare_modeling_dataset(
            election_data=election_data,
            demographics_data=demographics_data,
            target_variable=target_var,
            scale_features_flag=True
        )
        
        if not final_df.empty:
            print(f"  ✅ Dataset shape: {final_df.shape}")
            print(f"  ✅ Features: {len(feature_cols)}")
            print(f"  ✅ Target range: {final_df[target_var].min():.3f} to {final_df[target_var].max():.3f}")
            print(f"  ✅ Districts: {sorted(final_df['district_number'].unique())}")
            
            # Show sample of features
            print(f"  ✅ Sample features: {feature_cols[:5]}")
            
            # Show correlation with target
            if len(feature_cols) > 0:
                correlations = final_df[feature_cols + [target_var]].corr()[target_var].abs().sort_values(ascending=False)
                top_correlations = correlations.head(6)[1:6]  # Exclude target variable itself
                print(f"  ✅ Top 5 correlated features:")
                for feature, corr in top_correlations.items():
                    print(f"     {feature}: {corr:.3f}")
        else:
            print(f"  ❌ Failed to prepare dataset for {target_var}")
    
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    # Final dataset with democratic_share as target
    final_df, feature_cols = prepare_modeling_dataset(
        election_data=election_data,
        demographics_data=demographics_data,
        target_variable='democratic_share',
        scale_features_flag=True
    )
    
    if not final_df.empty:
        print(f"Final dataset ready for modeling:")
        print(f"  📊 Shape: {final_df.shape}")
        print(f"  🎯 Target: democratic_share")
        print(f"  🔢 Features: {len(feature_cols)}")
        print(f"  🗺️  Districts: {len(final_df['district_number'].unique())}")
        print(f"  📅 Years: {sorted(final_df['year'].unique())}")
        
        # Show feature categories
        census_features = [f for f in feature_cols if f.startswith('B') and f.endswith('E')]
        scaled_features = [f for f in feature_cols if f.endswith('_scaled')]
        lag_features = [f for f in feature_cols if 'lag' in f]
        
        print(f"\nFeature breakdown:")
        print(f"  📈 Census variables: {len(census_features)}")
        print(f"  📊 Scaled features: {len(scaled_features)}")
        print(f"  ⏰ Lag features: {len(lag_features)}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        display_cols = ['district_number', 'year', 'democratic_share', 'total_votes'] + feature_cols[:3]
        available_cols = [col for col in display_cols if col in final_df.columns]
        print(final_df[available_cols].head(3).to_string(index=False))
        
        # Save the prepared dataset
        output_file = 'data/processed/modeling_dataset.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\n💾 Dataset saved to: {output_file}")
        
        # Save feature list
        feature_file = 'data/processed/feature_list.txt'
        with open(feature_file, 'w') as f:
            f.write('\n'.join(feature_cols))
        print(f"📝 Feature list saved to: {feature_file}")
        
        print("\n✅ Feature engineering pipeline completed successfully!")
        print("   The dataset is now ready for model training.")
        
    else:
        print("❌ Failed to prepare final dataset")


def show_usage_examples():
    """Show usage examples for the feature engineering module."""
    
    print("\n" + "=" * 50)
    print("USAGE EXAMPLES")
    print("=" * 50)
    
    print("""
# Basic usage - prepare dataset for modeling
from src.features.build_features import prepare_modeling_dataset

# Load your data
election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')

# Prepare dataset
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    target_variable='democratic_share',
    scale_features_flag=True
)

# Use for model training
X = final_df[feature_cols]
y = final_df['democratic_share']

# Train your model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
""")
    
    print("""
# Advanced usage - custom feature selection
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    target_variable='total_votes',
    feature_cols=['B01001_001E', 'B19013_001E', 'B15003_001E'],  # Custom features
    scale_features_flag=False  # Don't scale features
)
""")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_feature_engineering()
    show_usage_examples()
