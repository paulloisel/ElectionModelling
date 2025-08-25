"""
Model Training Example

This script demonstrates the complete pipeline from feature engineering
to model training and analysis using the Election Modeling project.
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
from src.models.train_xgboost import (
    train_xgboost,
    feature_importance_analysis,
    evaluate_model,
    hyperparameter_tuning as xgb_hyperparameter_tuning
)
from src.models.train_elasticnet import (
    train_elasticnet,
    evaluate_model as elasticnet_evaluate,
    hyperparameter_tuning as elasticnet_hyperparameter_tuning
)


def load_data():
    """Load all required data for modeling."""
    
    print("Loading data...")
    
    # Load election data
    election_data = pd.read_csv('data/raw/wa_20241105_election_results.csv')
    print(f"  Election data: {election_data.shape}")
    
    # Load demographics data
    demographics_data = pd.read_csv('data/raw/census/wa_census_congressional_district_2020.csv')
    print(f"  Demographics data: {demographics_data.shape}")
    
    # Load registered voters data
    primary_voters = load_registered_voters_data(2024, "primary")
    general_voters = load_registered_voters_data(2024, "general")
    registered_voters_data = combine_registered_voters_data(primary_voters, general_voters)
    
    if not registered_voters_data.empty:
        print(f"  Registered voters data: {registered_voters_data.shape}")
    else:
        print("  No registered voters data available")
        registered_voters_data = None
    
    return election_data, demographics_data, registered_voters_data


def prepare_datasets(election_data, demographics_data, registered_voters_data):
    """Prepare datasets for different target variables."""
    
    print("\nPreparing modeling datasets...")
    
    # Define target variables to model
    target_variables = [
        'democratic_share',
        'turnout_rate', 
        'incumbent_share',
        'margin',
        'total_votes'
    ]
    
    datasets = {}
    
    for target_var in target_variables:
        print(f"  Preparing dataset for {target_var}...")
        
        try:
            final_df, feature_cols = prepare_modeling_dataset(
                election_data=election_data,
                demographics_data=demographics_data,
                registered_voters_data=registered_voters_data,
                target_variable=target_var,
                scale_features_flag=True
            )
            
            if not final_df.empty and target_var in final_df.columns:
                datasets[target_var] = {
                    'data': final_df,
                    'features': feature_cols,
                    'X': final_df[feature_cols].to_numpy(),
                    'y': final_df[target_var].to_numpy()
                }
                print(f"    ✅ Dataset shape: {final_df.shape}, Features: {len(feature_cols)}")
            else:
                print(f"    ❌ Failed to prepare dataset for {target_var}")
                
        except Exception as e:
            print(f"    ❌ Error preparing {target_var}: {e}")
    
    return datasets


def train_and_evaluate_models(datasets):
    """Train and evaluate models for each target variable."""
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    results = {}
    
    for target_var, dataset_info in datasets.items():
        print(f"\n--- Modeling {target_var} ---")
        
        X = dataset_info['X']
        y = dataset_info['y']
        feature_cols = dataset_info['features']
        
        # Split data (simple split for demonstration)
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        target_results = {}
        
        # Train XGBoost model
        print("  Training XGBoost model...")
        try:
            xgb_model = train_xgboost(X_train, y_train)
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
            xgb_importance = feature_importance_analysis(xgb_model, feature_cols)
            
            target_results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'importance': xgb_importance
            }
            
            print(f"    ✅ XGBoost R²: {xgb_metrics['r2']:.3f}, MSE: {xgb_metrics['mse']:.6f}")
            print(f"    Top features: {xgb_importance['feature'].iloc[0]} ({xgb_importance['importance'].iloc[0]:.3f})")
            
        except Exception as e:
            print(f"    ❌ XGBoost training failed: {e}")
        
        # Train ElasticNet model
        print("  Training ElasticNet model...")
        try:
            elasticnet_model = train_elasticnet(X_train, y_train, alpha=0.1, l1_ratio=0.5)
            elasticnet_metrics = elasticnet_evaluate(elasticnet_model, X_test, y_test)
            
            # Get feature coefficients
            coefficients = pd.DataFrame({
                'feature': feature_cols,
                'coefficient': elasticnet_model.coef_
            })
            coefficients = coefficients[coefficients['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)
            
            target_results['elasticnet'] = {
                'model': elasticnet_model,
                'metrics': elasticnet_metrics,
                'coefficients': coefficients
            }
            
            print(f"    ✅ ElasticNet R²: {elasticnet_metrics['r2']:.3f}, MSE: {elasticnet_metrics['mse']:.6f}")
            if not coefficients.empty:
                print(f"    Top feature: {coefficients['feature'].iloc[0]} (coef: {coefficients['coefficient'].iloc[0]:.6f})")
            
        except Exception as e:
            print(f"    ❌ ElasticNet training failed: {e}")
        
        results[target_var] = target_results
    
    return results


def analyze_results(results):
    """Analyze and summarize model results."""
    
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Create summary table
    summary_data = []
    
    for target_var, target_results in results.items():
        for model_name, model_results in target_results.items():
            metrics = model_results['metrics']
            summary_data.append({
                'Target': target_var,
                'Model': model_name,
                'R² Score': f"{metrics['r2']:.3f}",
                'MSE': f"{metrics['mse']:.6f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nModel Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Analyze feature importance across models
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    for target_var, target_results in results.items():
        print(f"\n--- {target_var.upper()} ---")
        
        if 'xgboost' in target_results:
            importance = target_results['xgboost']['importance']
            print("XGBoost Top 5 Features:")
            for i, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        if 'elasticnet' in target_results:
            coefficients = target_results['elasticnet']['coefficients']
            if not coefficients.empty:
                print("ElasticNet Top 5 Features:")
                for i, row in coefficients.head(5).iterrows():
                    print(f"  {row['feature']}: {row['coefficient']:.6f}")
    
    return summary_df


def save_results(results, summary_df):
    """Save model results and analysis."""
    
    print("\n" + "=" * 40)
    print("SAVING RESULTS")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("data/processed/model_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_df.to_csv(output_dir / "model_performance_summary.csv", index=False)
    print(f"✅ Model performance summary saved to: {output_dir / 'model_performance_summary.csv'}")
    
    # Save detailed results for each target
    for target_var, target_results in results.items():
        target_dir = output_dir / target_var
        target_dir.mkdir(exist_ok=True)
        
        for model_name, model_results in target_results.items():
            # Save feature importance/coefficients
            if 'importance' in model_results:
                model_results['importance'].to_csv(
                    target_dir / f"{model_name}_feature_importance.csv", 
                    index=False
                )
            
            if 'coefficients' in model_results:
                model_results['coefficients'].to_csv(
                    target_dir / f"{model_name}_coefficients.csv", 
                    index=False
                )
            
            # Save model
            try:
                model_path = target_dir / f"{model_name}_model.joblib"
                if model_name == 'xgboost':
                    from src.models.train_xgboost import save_model
                else:
                    from src.models.train_elasticnet import save_model
                
                save_model(model_results['model'], str(model_path))
                print(f"✅ {model_name} model for {target_var} saved to: {model_path}")
                
            except Exception as e:
                print(f"❌ Failed to save {model_name} model for {target_var}: {e}")


def demonstrate_hyperparameter_tuning(datasets):
    """Demonstrate hyperparameter tuning for one target variable."""
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("=" * 60)
    
    # Use democratic_share as example
    if 'democratic_share' in datasets:
        target_var = 'democratic_share'
        dataset_info = datasets[target_var]
        
        print(f"Tuning models for {target_var}...")
        
        X = dataset_info['X']
        y = dataset_info['y']
        feature_cols = dataset_info['features']
        
        # Split data
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # XGBoost hyperparameter tuning
        print("\nXGBoost Hyperparameter Tuning...")
        try:
            xgb_param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.3]
            }
            
            tuned_xgb = xgb_hyperparameter_tuning(X_train, y_train, xgb_param_grid)
            xgb_metrics = evaluate_model(tuned_xgb, X_test, y_test)
            
            print(f"✅ Tuned XGBoost R²: {xgb_metrics['r2']:.3f}")
            print(f"✅ Best parameters: {tuned_xgb.get_params()}")
            
        except Exception as e:
            print(f"❌ XGBoost tuning failed: {e}")
        
        # ElasticNet hyperparameter tuning
        print("\nElasticNet Hyperparameter Tuning...")
        try:
            elasticnet_param_grid = {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
            
            tuned_elasticnet = elasticnet_hyperparameter_tuning(X_train, y_train, elasticnet_param_grid)
            elasticnet_metrics = elasticnet_evaluate(tuned_elasticnet, X_test, y_test)
            
            print(f"✅ Tuned ElasticNet R²: {elasticnet_metrics['r2']:.3f}")
            print(f"✅ Best parameters: {tuned_elasticnet.get_params()}")
            
        except Exception as e:
            print(f"❌ ElasticNet tuning failed: {e}")


def main():
    """Main function to run the complete modeling pipeline."""
    
    print("=" * 80)
    print("ELECTION MODELING - COMPLETE PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Load data
    election_data, demographics_data, registered_voters_data = load_data()
    
    # Step 2: Prepare datasets
    datasets = prepare_datasets(election_data, demographics_data, registered_voters_data)
    
    if not datasets:
        print("❌ No datasets prepared successfully. Exiting.")
        return
    
    # Step 3: Train and evaluate models
    results = train_and_evaluate_models(datasets)
    
    # Step 4: Analyze results
    summary_df = analyze_results(results)
    
    # Step 5: Save results
    save_results(results, summary_df)
    
    # Step 6: Demonstrate hyperparameter tuning
    demonstrate_hyperparameter_tuning(datasets)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review model performance in data/processed/model_results/")
    print("2. Analyze feature importance for insights")
    print("3. Use trained models for predictions")
    print("4. Implement Mixed Linear Models for hierarchical analysis")


if __name__ == "__main__":
    main()
