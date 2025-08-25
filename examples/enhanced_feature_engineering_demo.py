"""
Enhanced Feature Engineering Demonstration

This script demonstrates the enhanced feature engineering pipeline
with turnout calculations and incumbent vote tracking.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.build_features import (
    prepare_modeling_dataset,
    load_registered_voters_data,
    combine_registered_voters_data
)


def demonstrate_enhanced_features():
    """Demonstrate the enhanced feature engineering with turnout and incumbent tracking."""
    
    print("=" * 80)
    print("ENHANCED FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    
    # Load election data
    election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
    print(f"   Election data: {election_data.shape}")
    
    # Load demographics data
    demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')
    print(f"   Demographics data: {demographics_data.shape}")
    
    # Load registered voters data
    print("\n2. Loading registered voters data...")
    primary_voters = load_registered_voters_data(2024, "primary")
    general_voters = load_registered_voters_data(2024, "general")
    
    if not primary_voters.empty:
        print(f"   Primary registered voters: {primary_voters.shape}")
    if not general_voters.empty:
        print(f"   General registered voters: {general_voters.shape}")
    
    # Combine registered voters data
    registered_voters_data = combine_registered_voters_data(primary_voters, general_voters)
    
    print("\n3. Enhanced feature engineering with turnout and incumbent tracking...")
    
    # Prepare dataset with enhanced features
    final_df, feature_cols = prepare_modeling_dataset(
        election_data=election_data,
        demographics_data=demographics_data,
        registered_voters_data=registered_voters_data,
        target_variable='democratic_share',
        scale_features_flag=True
    )
    
    if not final_df.empty:
        print(f"\n✅ Enhanced dataset created successfully!")
        print(f"   Shape: {final_df.shape}")
        print(f"   Features: {len(feature_cols)}")
        
        # Show new metrics
        print(f"\n📊 NEW ELECTION METRICS:")
        
        # Turnout metrics
        if 'turnout_rate' in final_df.columns:
            print(f"   Turnout Rate: {final_df['turnout_rate'].min():.3f} to {final_df['turnout_rate'].max():.3f}")
            print(f"   Average Turnout: {final_df['turnout_rate'].mean():.3f}")
        
        if 'registered_voters' in final_df.columns:
            print(f"   Registered Voters: {final_df['registered_voters'].min():,.0f} to {final_df['registered_voters'].max():,.0f}")
        
        # Incumbent metrics
        if 'incumbent_share' in final_df.columns:
            print(f"   Incumbent Vote Share: {final_df['incumbent_share'].min():.3f} to {final_df['incumbent_share'].max():.3f}")
            print(f"   Average Incumbent Share: {final_df['incumbent_share'].mean():.3f}")
        
        if 'challenger_share' in final_df.columns:
            print(f"   Challenger Vote Share: {final_df['challenger_share'].min():.3f} to {final_df['challenger_share'].max():.3f}")
        
        # Show district-level details
        print(f"\n🗺️  DISTRICT-LEVEL DETAILS:")
        display_cols = ['district_number', 'turnout_rate', 'democratic_share', 'incumbent_share', 'incumbent_party']
        available_cols = [col for col in display_cols if col in final_df.columns]
        
        for _, row in final_df.iterrows():
            district = row['district_number']
            turnout = row.get('turnout_rate', 0)
            dem_share = row.get('democratic_share', 0)
            inc_share = row.get('incumbent_share', 0)
            inc_party = row.get('incumbent_party', 'Unknown')
            
            print(f"   District {district}: Turnout={turnout:.3f}, Dem={dem_share:.3f}, Incumbent={inc_share:.3f} ({inc_party})")
        
        # Show feature correlations with new target variables
        print(f"\n📈 FEATURE CORRELATIONS WITH NEW TARGETS:")
        
        new_targets = ['turnout_rate', 'incumbent_share', 'challenger_share']
        for target in new_targets:
            if target in final_df.columns and len(feature_cols) > 0:
                correlations = final_df[feature_cols + [target]].corr()[target].abs().sort_values(ascending=False)
                top_correlations = correlations.head(4)[1:4]  # Exclude target itself
                print(f"\n   {target.upper()} correlations:")
                for feature, corr in top_correlations.items():
                    print(f"     {feature}: {corr:.3f}")
        
        # Save enhanced dataset
        output_file = 'data/processed/enhanced_modeling_dataset.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\n💾 Enhanced dataset saved to: {output_file}")
        
        # Show sample of enhanced data
        print(f"\n📋 SAMPLE OF ENHANCED DATASET:")
        sample_cols = ['district_number', 'turnout_rate', 'democratic_share', 'incumbent_share', 'incumbent_party', 'total_votes', 'registered_voters']
        available_sample_cols = [col for col in sample_cols if col in final_df.columns]
        print(final_df[available_sample_cols].head().to_string(index=False))
        
    else:
        print("❌ Failed to create enhanced dataset")


def demonstrate_multiple_targets():
    """Demonstrate modeling with multiple target variables including new metrics."""
    
    print("\n" + "=" * 60)
    print("MULTIPLE TARGET VARIABLES DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
    demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')
    registered_voters_data = load_registered_voters_data(2024, "general")
    
    # Test different target variables
    target_variables = [
        'democratic_share',      # Traditional target
        'turnout_rate',          # New: Voter turnout
        'incumbent_share',       # New: Incumbent performance
        'margin',                # Traditional: Victory margin
        'total_votes'            # Traditional: Total participation
    ]
    
    for target_var in target_variables:
        print(f"\n--- Modeling {target_var} ---")
        
        final_df, feature_cols = prepare_modeling_dataset(
            election_data=election_data,
            demographics_data=demographics_data,
            registered_voters_data=registered_voters_data,
            target_variable=target_var,
            scale_features_flag=True
        )
        
        if not final_df.empty and target_var in final_df.columns:
            print(f"  ✅ Dataset shape: {final_df.shape}")
            print(f"  ✅ Features: {len(feature_cols)}")
            print(f"  ✅ Target range: {final_df[target_var].min():.3f} to {final_df[target_var].max():.3f}")
            print(f"  ✅ Target mean: {final_df[target_var].mean():.3f}")
            
            # Show top correlations
            if len(feature_cols) > 0:
                correlations = final_df[feature_cols + [target_var]].corr()[target_var].abs().sort_values(ascending=False)
                top_correlation = correlations.head(2)[1:2]  # Get top correlation (excluding target itself)
                if not top_correlation.empty:
                    feature, corr = top_correlation.index[0], top_correlation.iloc[0]
                    print(f"  ✅ Top feature correlation: {feature} ({corr:.3f})")
        else:
            print(f"  ❌ Target variable {target_var} not available")


def show_usage_examples():
    """Show usage examples for the enhanced feature engineering."""
    
    print("\n" + "=" * 60)
    print("ENHANCED USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
# Enhanced feature engineering with turnout and incumbent tracking
from src.features.build_features import (
    prepare_modeling_dataset, 
    load_registered_voters_data,
    combine_registered_voters_data
)

# Load all data
election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')

# Load registered voters data
primary_voters = load_registered_voters_data(2024, "primary")
general_voters = load_registered_voters_data(2024, "general")
registered_voters_data = combine_registered_voters_data(primary_voters, general_voters)

# Prepare enhanced dataset
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='turnout_rate',  # New target: voter turnout
    scale_features_flag=True
)

# Model turnout prediction
X = final_df[feature_cols]
y = final_df['turnout_rate']
""")
    
    print("""
# Model incumbent performance
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='incumbent_share',  # New target: incumbent vote share
    scale_features_flag=True
)

# Model incumbent vote prediction
X = final_df[feature_cols]
y = final_df['incumbent_share']
""")


if __name__ == "__main__":
    # Run enhanced demonstration
    demonstrate_enhanced_features()
    demonstrate_multiple_targets()
    show_usage_examples()
