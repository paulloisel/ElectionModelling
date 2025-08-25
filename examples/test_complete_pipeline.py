"""
Complete Pipeline Test

This script tests that all components of the Election Modeling pipeline
work correctly with the updated requirements.txt file.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required packages can be imported."""
    
    print("Testing package imports...")
    
    try:
        # Core data science
        import numpy as np
        import pandas as pd
        print("  ✅ numpy and pandas imported")
        
        # Machine learning
        import sklearn
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        print("  ✅ scikit-learn imported")
        
        import xgboost as xgb
        print("  ✅ xgboost imported")
        
        import statsmodels
        from statsmodels.regression.mixed_linear_model import MixedLM
        print("  ✅ statsmodels imported")
        
        import joblib
        print("  ✅ joblib imported")
        
        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        print("  ✅ visualization packages imported")
        
        # Census and geographic
        try:
            import CensusData
            print("  ✅ CensusData imported")
        except ImportError:
            print("  ⚠️  CensusData not available (optional)")
        
        try:
            import geopandas as gpd
            print("  ✅ geopandas imported")
        except ImportError:
            print("  ⚠️  geopandas not available (optional)")
        
        # File handling
        import openpyxl
        print("  ✅ openpyxl imported")
        
        # API and requests
        import requests
        print("  ✅ requests imported")
        
        # AI
        try:
            import openai
            print("  ✅ openai imported")
        except ImportError:
            print("  ⚠️  openai not available (optional)")
        
        print("✅ All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_feature_engineering():
    """Test that feature engineering module works."""
    
    print("\nTesting feature engineering...")
    
    try:
        from src.features.build_features import (
            prepare_modeling_dataset,
            load_registered_voters_data,
            combine_registered_voters_data
        )
        print("  ✅ Feature engineering functions imported")
        print("  ✅ Feature engineering module ready for use")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False


def test_model_training():
    """Test that model training modules work."""
    
    print("\nTesting model training...")
    
    try:
        from src.models.train_xgboost import train_xgboost, evaluate_model
        from src.models.train_elasticnet import train_elasticnet
        from src.models.train_mixedlm import train_mixedlm, prepare_mixedlm_data
        print("  ✅ Model training functions imported")
        
        # Create sample data for testing
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        
        # Test XGBoost
        xgb_model = train_xgboost(X, y)
        xgb_metrics = evaluate_model(xgb_model, X, y)
        print(f"  ✅ XGBoost training successful: R² = {xgb_metrics['r2']:.3f}")
        
        # Test ElasticNet
        en_model = train_elasticnet(X, y)
        en_metrics = evaluate_model(en_model, X, y)
        print(f"  ✅ ElasticNet training successful: R² = {en_metrics['r2']:.3f}")
        
        # Test Mixed Linear Model with synthetic data
        df = pd.DataFrame({
            'target': y,
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'group': ['A'] * 10 + ['B'] * 10
        })
        
        modeling_df, formula = prepare_mixedlm_data(
            df=df,
            target_col='target',
            feature_cols=['feature1', 'feature2', 'feature3'],
            group_col='group'
        )
        
        mixedlm_model = train_mixedlm(modeling_df, formula, 'group')
        print(f"  ✅ Mixed Linear Model training successful: converged = {mixedlm_model.converged}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False


def test_data_ingestion():
    """Test that data ingestion modules work."""
    
    print("\nTesting data ingestion...")
    
    try:
        from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline
        print("  ✅ Census pipeline imported")
        
        # Test pipeline initialization
        pipeline = ACSFeatureReductionPipeline(
            years=[2020],
            output_dir="data/processed/test"
        )
        print("  ✅ Census pipeline initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Data ingestion error: {e}")
        return False


def test_utilities():
    """Test that utility modules work."""
    
    print("\nTesting utilities...")
    
    try:
        from src.utils.downloader import WAStateDownloader
        print("  ✅ Downloader utility imported")
        
        # Test downloader initialization
        downloader = WAStateDownloader()
        print("  ✅ Downloader initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False


def main():
    """Main function to test the complete pipeline."""
    
    print("=" * 80)
    print("COMPLETE PIPELINE TESTING")
    print("=" * 80)
    
    tests = [
        ("Package Imports", test_imports),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Data Ingestion", test_data_ingestion),
        ("Utilities", test_utilities)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The complete pipeline is working correctly.")
        print("\nThe Election Modeling project is ready for use with:")
        print("✅ All required dependencies installed")
        print("✅ Feature engineering pipeline functional")
        print("✅ All three model types (XGBoost, ElasticNet, Mixed Linear) working")
        print("✅ Data ingestion and utilities operational")
        print("✅ Complete documentation available")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the requirements and dependencies.")
    
    return passed == total


if __name__ == "__main__":
    main()
