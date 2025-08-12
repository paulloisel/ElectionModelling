"""Visualize Washington congressional district census data trends."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import math

def main():
    # Find the most recent analysis files
    analysis_files = glob('wa_congressional_analysis_*.csv')
    metadata_files = glob('wa_congressional_variables_*.csv')
    
    if not analysis_files or not metadata_files:
        print("No analysis files found!")
        return
    
    # Load the most recent files
    data = pd.read_csv(sorted(analysis_files)[-1])
    metadata = pd.read_csv(sorted(metadata_files)[-1])
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    
    # 1. Summary Statistics
    print("\nSummary Statistics by District:")
    summary = data.groupby('district').agg(['mean', 'std']).round(2)
    print(summary)
    
    # 2. District Comparison - Selected Variables
    # Select a subset of interesting variables
    selected_vars = [col for col in data.columns if col not in ['district', 'year']][:5]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    district_means = data.groupby('district')[selected_vars].mean()
    district_means.plot(kind='bar', ax=ax)
    ax.set_title('Average Values by Congressional District (Selected Variables)')
    ax.set_xlabel('District')
    ax.set_ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('wa_congressional_comparison.png')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    selected_cols = list(numeric_cols)[:10]  # Select first 10 variables for visibility
    sns.heatmap(data[selected_cols].corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Variable Correlation Heatmap (Top 10 Variables)')
    plt.tight_layout()
    plt.savefig('wa_congressional_correlations.png')
    plt.close()
    
    # 4. Time Series for Selected Variables
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, var in enumerate(selected_vars[:6]):  # Plot first 6 variables
        for district in data['district'].unique():
            district_data = data[data['district'] == district]
            axes[i].plot(district_data['year'], district_data[var], label=district)
        axes[i].set_title(f'Trends for {var}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Value')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('wa_congressional_trends.png')
    plt.close()
    
    # Print variable descriptions
    print("\nSelected Variables Description:")
    for var in selected_vars:
        var_meta = metadata[metadata['name'] == var]
        if not var_meta.empty:
            print(f"\n{var}:")
            print(f"Label: {var_meta['label'].iloc[0]}")
            if 'concept' in var_meta.columns:
                print(f"Concept: {var_meta['concept'].iloc[0]}")
    
    print("\nVisualizations saved as:")
    print("- wa_congressional_comparison.png")
    print("- wa_congressional_correlations.png")
    print("- wa_congressional_trends.png")

if __name__ == "__main__":
    main()