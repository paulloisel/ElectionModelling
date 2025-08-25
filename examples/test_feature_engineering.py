"""
Test script for the feature engineering module.

This script demonstrates how to use the feature engineering functions
with real election and demographics data.
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.build_features import (
    join_election_demographics,
    create_lag_features,
    engineer_demographic_features,
    create_interaction_features,
    scale_features,
    prepare_modeling_dataset
)


def test_feature_engineering():
    """Test the feature engineering pipeline with real data."""
    
    print("=" * 60)
    print("TESTING FEATURE ENGINEERING MODULE")
    print("=" * 60)
    
    # Load sample data
    print("\n1. Loading sample data...")
    
    # Load election data
    election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
    print(f"   Election data shape: {election_data.shape}")
    print(f"   Election data columns: {election_data.columns.tolist()}")
    
    # Load demographics data
    demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')
    print(f"   Demographics data shape: {demographics_data.shape}")
    print(f"   Demographics data columns: {demographics_data.columns.tolist()}")
    
    # Test individual functions
    print("\n2. Testing individual feature engineering functions...")
    
    # Test data joining
    print("\n   Testing data joining...")
    joined_df = join_election_demographics(election_data, demographics_data)
    print(f"   Joined data shape: {joined_df.shape}")
    if not joined_df.empty:
        print(f"   Districts included: {sorted(joined_df['district_number'].unique())}")
        print(f"   Sample columns: {joined_df.columns.tolist()[:10]}")
    
    if joined_df.empty:
        print("   ❌ Data joining failed - stopping test")
        return
    
    # Test lag features
    print("\n   Testing lag features...")
    # Add year column for lag features
    joined_df['year'] = 2024
    lag_df = create_lag_features(joined_df, 'democratic_share', lag_periods=[1, 2])
    print(f"   Lag features created. Shape: {lag_df.shape}")
    lag_cols = [col for col in lag_df.columns if 'lag' in col]
    print(f"   Lag columns: {lag_cols}")
    
    # Test demographic feature engineering
    print("\n   Testing demographic feature engineering...")
    engineered_df = engineer_demographic_features(lag_df)
    print(f"   Engineered features added. Shape: {engineered_df.shape}")
    engineered_cols = [col for col in engineered_df.columns if any(suffix in col for suffix in ['_pct', '_log', '_range'])]
    print(f"   Engineered columns: {engineered_cols[:5]}...")
    
    # Test interaction features
    print("\n   Testing interaction features...")
    interaction_df = create_interaction_features(engineered_df)
    print(f"   Interaction features added. Shape: {interaction_df.shape}")
    interaction_cols = [col for col in interaction_df.columns if '_x_' in col]
    print(f"   Interaction columns: {interaction_cols}")
    
    # Test feature scaling
    print("\n   Testing feature scaling...")
    # Get some census columns for scaling
    census_cols = [col for col in interaction_df.columns if col.startswith('B') and col.endswith('E')]
    if census_cols:
        scaled_df = scale_features(interaction_df, census_cols[:5])  # Scale first 5 census variables
        print(f"   Features scaled. Shape: {scaled_df.shape}")
        scaled_cols = [col for col in scaled_df.columns if col.endswith('_scaled')]
        print(f"   Scaled columns: {scaled_cols}")
    
    # Test complete pipeline
    print("\n3. Testing complete modeling dataset preparation...")
    
    final_df, feature_cols = prepare_modeling_dataset(
        election_data=election_data,
        demographics_data=demographics_data,
        target_variable='democratic_share',
        scale_features_flag=True
    )
    
    print(f"   Final dataset shape: {final_df.shape}")
    print(f"   Number of features: {len(feature_cols)}")
    print(f"   Target variable: democratic_share")
    
    if not final_df.empty:
        print(f"   Sample features: {feature_cols[:10]}")
        print(f"   Target variable range: {final_df['democratic_share'].min():.3f} to {final_df['democratic_share'].max():.3f}")
        
        # Show sample of final dataset
        print(f"\n   Sample of final dataset:")
        sample_cols = ['district_number', 'democratic_share', 'total_votes'] + feature_cols[:3]
        available_cols = [col for col in sample_cols if col in final_df.columns]
        print(final_df[available_cols].head())
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING TEST COMPLETED")
    print("=" * 60)


def test_multiple_years():
    """Test with multiple years of data."""
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE YEARS OF DATA")
    print("=" * 60)
    
    # Load multiple years of census data
    years = [2018, 2019, 2020]
    all_demographics = []
    
    for year in years:
        try:
            census_file = f'data/raw/census/wa_census_congressional_district_{year}.csv'
            if os.path.exists(census_file):
                df = pd.read_csv(census_file)
                all_demographics.append(df)
                print(f"   Loaded {year} census data: {df.shape}")
        except Exception as e:
            print(f"   Could not load {year} data: {e}")
    
    if not all_demographics:
        print("   No census data found for multiple years")
        return
    
    # Combine multiple years
    combined_demographics = pd.concat(all_demographics, ignore_index=True)
    print(f"   Combined demographics shape: {combined_demographics.shape}")
    
    # Load election data (we'll use 2024 as example)
    election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
    
    # Test with combined data
    final_df, feature_cols = prepare_modeling_dataset(
        election_data=election_data,
        demographics_data=combined_demographics,
        target_variable='democratic_share',
        scale_features_flag=True
    )
    
    print(f"   Multi-year dataset shape: {final_df.shape}")
    print(f"   Features: {len(feature_cols)}")
    
    if not final_df.empty:
        print(f"   Years in dataset: {sorted(final_df['year'].unique())}")


if __name__ == "__main__":
    # Run tests
    test_feature_engineering()
    test_multiple_years()
