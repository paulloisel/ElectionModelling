# Election Modeling Pipeline Documentation

## 🏗️ **COMPLETE PIPELINE OVERVIEW**

This document provides a comprehensive guide to the Election Modeling pipeline, from data ingestion through feature engineering to model training.

## 📊 **PIPELINE ARCHITECTURE**

```
Data Sources
    ├── Election Data (Washington State)
    ├── Census Demographics (ACS)
    └── Registered Voters Data

Data Ingestion
    ├── WA Results Loader
    ├── Census Pipeline
    └── Registered Voters Loader

Feature Engineering
    ├── Data Joining
    ├── Turnout Calculation
    ├── Incumbent Tracking
    ├── Lag Features
    ├── Demographic Engineering
    └── Feature Scaling

Model Training
    ├── XGBoost (Gradient Boosting)
    ├── ElasticNet (Linear with Regularization)
    └── Mixed Linear Models (Hierarchical)
```

## 🚀 **QUICK START GUIDE**

### **1. Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd ElectionModelling

# Create conda environment
conda env create -f environment.yml
conda activate election-modeling

# Verify installation
python -c "import pandas, xgboost, statsmodels; print('Setup complete!')"
```

### **2. Download Data**
```bash
# Download Washington State election data
python -c "
from src.utils.downloader import WAStateDownloader
downloader = WAStateDownloader()
downloader.download_all_even_years_data()
"
```

### **3. Run Feature Engineering**
```bash
# Basic feature engineering
python examples/feature_engineering_demo.py

# Enhanced features with turnout and incumbent tracking
python examples/enhanced_feature_engineering_demo.py
```

### **4. Train Models**
```bash
# Example model training (see detailed examples below)
python examples/model_training_example.py
```

## 📁 **DATA SOURCES**

### **Election Data**
- **Source**: Washington State Secretary of State
- **Format**: XLSX and CSV files
- **Years**: 2012-2024
- **Content**: Congressional district results, vote counts, percentages

### **Demographics Data**
- **Source**: American Community Survey (ACS)
- **Format**: Census API data
- **Years**: 2010-2023
- **Content**: Income, education, age, housing, employment, poverty

### **Registered Voters Data**
- **Source**: Washington State Secretary of State
- **Format**: XLSX files
- **Years**: 2014-2024
- **Content**: Registered voter counts by district

## 🔧 **CORE COMPONENTS**

### **1. Data Ingestion (`src/ingest/`)**

#### **WA Results Loader**
```python
from src.ingest.wa_results_loader import WAResultsLoader

# Load election results
loader = WAResultsLoader()
results = loader.load_election_results('data/raw/wa_20241105_election_results.csv')
```

#### **Census Pipeline**
```python
from src.ingest.censuspipeline.pipeline import ACSFeatureReductionPipeline

# Initialize pipeline
pipeline = ACSFeatureReductionPipeline(
    years=[2020, 2021, 2022],
    output_dir="data/processed/census"
)

# Load metadata and select variables
metadata = pipeline.load_metadata()
selected = pipeline.select_variables(
    keywords=["income", "education", "employment"],
    openai_top_k=20
)
```

#### **Registered Voters Loader**
```python
from src.features.build_features import load_registered_voters_data

# Load registered voters data
primary_voters = load_registered_voters_data(2024, "primary")
general_voters = load_registered_voters_data(2024, "general")
```

### **2. Feature Engineering (`src/features/`)**

#### **Complete Pipeline**
```python
from src.features.build_features import prepare_modeling_dataset

# Prepare complete dataset
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='turnout_rate',
    scale_features_flag=True
)
```

#### **Individual Functions**
```python
from src.features.build_features import (
    join_election_demographics,
    create_lag_features,
    engineer_demographic_features,
    create_interaction_features,
    scale_features
)

# Join election and demographics data
joined_df = join_election_demographics(election_data, demographics_data, registered_voters_data)

# Create lag features
lag_df = create_lag_features(joined_df, 'democratic_share', lag_periods=[1, 2, 4])

# Engineer demographic features
engineered_df = engineer_demographic_features(lag_df)

# Create interaction features
interaction_df = create_interaction_features(engineered_df)

# Scale features
scaled_df = scale_features(interaction_df, feature_cols)
```

### **3. Model Training (`src/models/`)**

#### **XGBoost Model**
```python
from src.models.train_xgboost import (
    train_xgboost,
    feature_importance_analysis,
    evaluate_model
)

# Train model
model = train_xgboost(X_train, y_train)

# Analyze feature importance
importance_df = feature_importance_analysis(model, feature_cols)
print("Top demographic predictors:")
print(importance_df.head(10))

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"R² Score: {metrics['r2']:.3f}")
```

#### **ElasticNet Model**
```python
from src.models.train_elasticnet import (
    train_elasticnet,
    hyperparameter_tuning,
    evaluate_model
)

# Train with hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}
model = hyperparameter_tuning(X_train, y_train, param_grid)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
```

## 📈 **TARGET VARIABLES**

### **Available Targets**
1. **`democratic_share`**: Democratic vote percentage (0.0 to 1.0)
2. **`turnout_rate`**: Voter turnout rate (0.363 to 0.435)
3. **`incumbent_share`**: Incumbent vote percentage (0.520 to 0.839)
4. **`total_votes`**: Total votes cast in district
5. **`margin`**: Margin of victory (0.173 to 0.982)

### **Feature Variables**
- **20 Census Variables**: Raw demographic counts
- **20 Scaled Features**: Standardized versions
- **Lag Features**: Historical patterns (when multiple years available)
- **Interaction Features**: Cross-variable interactions

## 🔍 **USAGE EXAMPLES**

### **Example 1: Turnout Prediction**
```python
# Load and prepare data
from src.features.build_features import prepare_modeling_dataset
from src.models.train_xgboost import train_xgboost, feature_importance_analysis

# Prepare dataset for turnout prediction
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='turnout_rate'
)

# Train XGBoost model
X = final_df[feature_cols].to_numpy()
y = final_df['turnout_rate'].to_numpy()
model = train_xgboost(X, y)

# Analyze which demographics predict turnout
importance_df = feature_importance_analysis(model, feature_cols)
print("Demographics that predict voter turnout:")
print(importance_df.head(10))
```

### **Example 2: Incumbent Performance Analysis**
```python
# Prepare dataset for incumbent performance
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='incumbent_share'
)

# Train ElasticNet for interpretable results
from src.models.train_elasticnet import train_elasticnet

model = train_elasticnet(X, y, alpha=0.1, l1_ratio=0.5)

# Get feature coefficients
coefficients = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_
})
coefficients = coefficients[coefficients['coefficient'] != 0].sort_values('coefficient', key=abs, ascending=False)
print("Demographics that affect incumbent performance:")
print(coefficients.head(10))
```

### **Example 3: Democratic Vote Share Prediction**
```python
# Prepare dataset for Democratic vote share
final_df, feature_cols = prepare_modeling_dataset(
    election_data=election_data,
    demographics_data=demographics_data,
    registered_voters_data=registered_voters_data,
    target_variable='democratic_share'
)

# Train both models for comparison
from src.models.train_xgboost import train_xgboost
from src.models.train_elasticnet import train_elasticnet

# XGBoost for non-linear relationships
xgboost_model = train_xgboost(X, y)

# ElasticNet for interpretable linear relationships
elasticnet_model = train_elasticnet(X, y)

# Compare feature importance
xgboost_importance = feature_importance_analysis(xgboost_model, feature_cols)
elasticnet_coefs = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': elasticnet_model.coef_
})

print("XGBoost feature importance:")
print(xgboost_importance.head(5))
print("\nElasticNet coefficients:")
print(elasticnet_coefs[elasticnet_coefs['coefficient'] != 0].head(5))
```

## 📊 **OUTPUT FILES**

### **Processed Data**
- **`data/processed/modeling_dataset.csv`**: Basic modeling dataset
- **`data/processed/enhanced_modeling_dataset.csv`**: Enhanced dataset with turnout and incumbent tracking
- **`data/processed/feature_list.txt`**: List of feature column names

### **Model Outputs**
- **Trained Models**: Saved as joblib files
- **Feature Importance**: CSV files with feature rankings
- **Model Metrics**: Performance evaluation results

## 🧪 **TESTING**

### **Run Tests**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_feature_engineering.py
python -m pytest tests/test_model_training.py
```

### **Test Coverage**
- **Feature Engineering**: Data joining, feature creation, scaling
- **Model Training**: XGBoost, ElasticNet training and evaluation
- **Data Pipeline**: End-to-end workflow testing

## 📚 **ADDITIONAL RESOURCES**

### **Documentation Files**
- **`README.md`**: Project overview and setup
- **`FEATURE_ENGINEERING_SUMMARY.md`**: Detailed feature engineering documentation
- **`examples/README.md`**: Example scripts documentation
- **`tests/README.md`**: Testing documentation

### **Example Scripts**
- **`examples/feature_engineering_demo.py`**: Basic feature engineering
- **`examples/enhanced_feature_engineering_demo.py`**: Enhanced features with turnout tracking
- **`examples/census_example.py`**: Census pipeline usage
- **`examples/wa_congressional_analysis.py`**: Washington congressional analysis

### **Notebooks**
- **`notebooks/01_turnout_model_walkthrough.ipynb`**: Interactive walkthrough (needs implementation)

## 🔄 **WORKFLOW SUMMARY**

1. **Data Collection**: Download election and demographics data
2. **Data Processing**: Clean and prepare raw data
3. **Feature Engineering**: Create modeling features with turnout and incumbent tracking
4. **Model Training**: Train XGBoost and ElasticNet models
5. **Analysis**: Analyze feature importance and model performance
6. **Interpretation**: Understand demographic factors affecting election outcomes

## 🎯 **NEXT STEPS**

1. **Implement Mixed Linear Models**: Complete the hierarchical modeling component
2. **Expand Data Sources**: Add more states and election years
3. **Create Interactive Notebooks**: Develop Jupyter notebooks for analysis
4. **Model Deployment**: Create API endpoints for real-time predictions
5. **Performance Optimization**: Optimize for larger datasets and faster training

This pipeline provides a complete framework for election modeling, from raw data to actionable insights about the relationship between demographics and election outcomes.
