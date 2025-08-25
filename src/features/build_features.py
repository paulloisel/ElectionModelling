"""
Feature Engineering Module

Builds features for election modeling including joins, lag features, and demographics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import re
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_district_number(district_name: str) -> int:
    """
    Extract district number from congressional district name.
    
    Args:
        district_name: Congressional district name string
        
    Returns:
        District number as integer
    """
    if pd.isna(district_name):
        return None
    
    # Extract district number using regex
    match = re.search(r'District (\d+)', district_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def process_election_data(election_df: pd.DataFrame, registered_voters_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Process raw election data to extract district-level metrics.
    
    Args:
        election_df: Raw election results DataFrame
        registered_voters_df: Optional DataFrame with registered voters data for turnout calculation
        
    Returns:
        Processed election DataFrame with district-level metrics
    """
    # Filter for congressional district races
    congressional_races = election_df[
        election_df['Race'].str.contains('Congressional District', na=False)
    ].copy()
    
    if congressional_races.empty:
        logger.warning("No congressional district races found in election data")
        return pd.DataFrame()
    
    # Extract district number
    congressional_races['district_number'] = congressional_races['Race'].apply(extract_district_number)
    
    # Get registered voters data if provided
    registered_voters_by_district = {}
    if registered_voters_df is not None:
        try:
            # Extract registered voters by district
            if 'RaceName' in registered_voters_df.columns:
                turnout_data = registered_voters_df[registered_voters_df['RaceName'] == 'Turnout']
                if not turnout_data.empty:
                    registered_voters_by_district = turnout_data.groupby('DistrictNumber')['Votes'].sum().to_dict()
                    logger.info(f"Loaded registered voters data for {len(registered_voters_by_district)} districts")
        except Exception as e:
            logger.warning(f"Could not process registered voters data: {e}")
    
    # Define incumbent information (based on 2024 election)
    incumbent_info = {
        1: {'name': 'Suzan DelBene', 'party': 'Democratic', 'incumbent': True},
        2: {'name': 'Rick Larsen', 'party': 'Democratic', 'incumbent': True},
        3: {'name': 'Marie Gluesenkamp Perez', 'party': 'Democratic', 'incumbent': True},
        4: {'name': 'Dan Newhouse', 'party': 'Republican', 'incumbent': True},
        5: {'name': 'Cathy McMorris Rodgers', 'party': 'Republican', 'incumbent': False},  # Retired
        6: {'name': 'Derek Kilmer', 'party': 'Democratic', 'incumbent': False},  # Retired
        7: {'name': 'Pramila Jayapal', 'party': 'Democratic', 'incumbent': True},
        8: {'name': 'Kim Schrier', 'party': 'Democratic', 'incumbent': True},
        9: {'name': 'Adam Smith', 'party': 'Democratic', 'incumbent': True},
        10: {'name': 'Marilyn Strickland', 'party': 'Democratic', 'incumbent': True}
    }
    
    # Calculate district-level metrics
    district_metrics = []
    
    for district in congressional_races['district_number'].unique():
        if pd.isna(district):
            continue
            
        district_data = congressional_races[congressional_races['district_number'] == district]
        
        # Calculate total votes in district
        total_votes = district_data['Votes'].sum()
        
        # Get Democratic and Republican votes based on candidate names and party patterns
        dem_votes = 0
        rep_votes = 0
        incumbent_votes = 0
        challenger_votes = 0
        
        # Get incumbent info for this district
        incumbent_data = incumbent_info.get(district, {})
        incumbent_name = incumbent_data.get('name', '').lower()
        incumbent_party = incumbent_data.get('party', '')
        is_incumbent_race = incumbent_data.get('incumbent', False)
        
        for _, row in district_data.iterrows():
            candidate = row['Candidate'].lower()
            votes = row['Votes']
            party = row.get('Party', '').lower()
            
            # Skip write-ins
            if 'write-in' in candidate:
                continue
            
            # Determine party and incumbent status
            is_democratic = False
            is_republican = False
            is_incumbent = False
            
            # Check party based on candidate name patterns
            if any(dem_indicator in candidate for dem_indicator in [
                'delbene', 'larsen', 'jayapal', 'smith', 'strickland', 'conroy', 'randall', 'schrier'
            ]):
                is_democratic = True
            elif any(rep_indicator in candidate for rep_indicator in [
                'brewer', 'hart', 'kent', 'sessler', 'newhouse', 'baumgartner', 'macewen', 'alexander', 'goers', 'hewett'
            ]):
                is_republican = True
            
            # Check party from Party column if available
            if 'democratic' in party:
                is_democratic = True
            elif 'republican' in party or 'gop' in party:
                is_republican = True
            
            # Check if this is the incumbent
            if is_incumbent_race and incumbent_name and any(name_part in candidate for name_part in incumbent_name.split()):
                is_incumbent = True
            
            # Accumulate votes
            if is_democratic:
                dem_votes += votes
            if is_republican:
                rep_votes += votes
            
            if is_incumbent:
                incumbent_votes += votes
            else:
                challenger_votes += votes
        
        # Calculate turnout if registered voters data is available
        registered_voters = registered_voters_by_district.get(district, 0)
        turnout_rate = total_votes / registered_voters if registered_voters > 0 else 0
        
        # Calculate metrics
        metrics = {
            'district_number': district,
            'total_votes': total_votes,
            'registered_voters': registered_voters,
            'turnout_rate': turnout_rate,
            'democratic_votes': dem_votes,
            'republican_votes': rep_votes,
            'democratic_share': dem_votes / total_votes if total_votes > 0 else 0,
            'republican_share': rep_votes / total_votes if total_votes > 0 else 0,
            'margin': abs(dem_votes - rep_votes) / total_votes if total_votes > 0 else 0,
            'winner': 'Democratic' if dem_votes > rep_votes else 'Republican',
            'incumbent_votes': incumbent_votes,
            'challenger_votes': challenger_votes,
            'incumbent_share': incumbent_votes / total_votes if total_votes > 0 else 0,
            'challenger_share': challenger_votes / total_votes if total_votes > 0 else 0,
            'incumbent_party': incumbent_party,
            'is_incumbent_race': is_incumbent_race
        }
        
        district_metrics.append(metrics)
    
    return pd.DataFrame(district_metrics)


def process_demographics_data(demographics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw demographics data to extract district number and clean data.
    
    Args:
        demographics_df: Raw demographics DataFrame
        
    Returns:
        Processed demographics DataFrame
    """
    processed_df = demographics_df.copy()
    
    # Extract district number from NAME column
    processed_df['district_number'] = processed_df['NAME'].apply(extract_district_number)
    
    # Remove rows where district number couldn't be extracted
    processed_df = processed_df.dropna(subset=['district_number'])
    
    # Convert district number to integer
    processed_df['district_number'] = processed_df['district_number'].astype(int)
    
    # Drop the NAME column as it's no longer needed
    processed_df = processed_df.drop('NAME', axis=1)
    
    return processed_df


def load_registered_voters_data(year: int, election_type: str = "general") -> pd.DataFrame:
    """
    Load registered voters data for a specific year and election type.
    
    Args:
        year: Election year
        election_type: "primary" or "general"
        
    Returns:
        DataFrame with registered voters data
    """
    try:
        if election_type == "primary":
            file_path = f"data/raw/wa_{year}_pri_registered_voters.xlsx"
        else:
            file_path = f"data/raw/wa_{year}_gen_registered_voters.xlsx"
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            logger.info(f"Loaded {election_type} registered voters data for {year}: {df.shape}")
            return df
        else:
            logger.warning(f"Registered voters file not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading registered voters data: {e}")
        return pd.DataFrame()


def combine_registered_voters_data(primary_data: pd.DataFrame, general_data: pd.DataFrame) -> pd.DataFrame:
    """
    Combine primary and general election registered voters data.
    Uses general election data as the base, falls back to primary if general not available.
    
    Args:
        primary_data: Primary election registered voters data
        general_data: General election registered voters data
        
    Returns:
        Combined registered voters DataFrame
    """
    if not general_data.empty:
        logger.info("Using general election registered voters data")
        return general_data
    elif not primary_data.empty:
        logger.info("Using primary election registered voters data (general not available)")
        return primary_data
    else:
        logger.warning("No registered voters data available")
        return pd.DataFrame()


def join_election_demographics(
    election_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    registered_voters_df: pd.DataFrame = None,
    join_key: str = "district_number"
) -> pd.DataFrame:
    """
    Join election results with demographic data.
    
    Args:
        election_df: Election results DataFrame
        demographics_df: Demographics DataFrame
        registered_voters_df: Optional DataFrame with registered voters data for turnout calculation
        join_key: Column to join on (default: district_number)
        
    Returns:
        Joined DataFrame
    """
    logger.info("Joining election and demographics data...")
    
    # Process election data with registered voters for turnout calculation
    processed_election = process_election_data(election_df, registered_voters_df)
    
    if processed_election.empty:
        logger.error("No valid election data to join")
        return pd.DataFrame()
    
    # Process demographics data
    processed_demographics = process_demographics_data(demographics_df)
    
    if processed_demographics.empty:
        logger.error("No valid demographics data to join")
        return pd.DataFrame()
    
    # Join the datasets
    joined_df = processed_election.merge(
        processed_demographics,
        on=join_key,
        how='inner'
    )
    
    logger.info(f"Successfully joined data. Shape: {joined_df.shape}")
    logger.info(f"Districts included: {sorted(joined_df[join_key].unique())}")
    
    return joined_df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_periods: List[int] = [1, 2, 4],
    group_col: str = "district_number",
    time_col: str = "year"
) -> pd.DataFrame:
    """
    Create lag features for time series analysis.
    
    Args:
        df: Input DataFrame
        target_col: Target variable column
        lag_periods: List of lag periods to create
        group_col: Column to group by for lags
        time_col: Time column for sorting
        
    Returns:
        DataFrame with lag features
    """
    logger.info(f"Creating lag features for {target_col}...")
    
    # Sort by group and time
    df_sorted = df.sort_values([group_col, time_col]).copy()
    
    # Create lag features for each group
    for lag in lag_periods:
        lag_col = f"{target_col}_lag_{lag}"
        df_sorted[lag_col] = df_sorted.groupby(group_col)[target_col].shift(lag)
        logger.info(f"Created {lag_col}")
    
    return df_sorted


def engineer_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer demographic features from raw data.
    
    Args:
        df: Input DataFrame with demographic variables
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering demographic features...")
    
    engineered_df = df.copy()
    
    # Get census variable columns (those starting with B and ending with E)
    census_cols = [col for col in df.columns if re.match(r'^B\d+.*E$', col)]
    
    if not census_cols:
        logger.warning("No census variables found for feature engineering")
        return engineered_df
    
    logger.info(f"Found {len(census_cols)} census variables for feature engineering")
    
    # Create demographic ratios and percentages
    for col in census_cols:
        # Skip if column contains mostly zeros or NaN
        if engineered_df[col].isna().sum() > len(engineered_df) * 0.5:
            continue
            
        # Create percentage features for population-based variables
        if 'B01001' in col:  # Population variables
            # Calculate as percentage of total population
            total_pop_cols = [c for c in census_cols if 'B01001_001E' in c]
            if total_pop_cols:
                total_pop = engineered_df[total_pop_cols[0]]
                if total_pop.sum() > 0:
                    engineered_df[f"{col}_pct"] = engineered_df[col] / total_pop * 100
        
        # Create income-based features
        if 'B19013' in col:  # Median income
            engineered_df[f"{col}_log"] = np.log(engineered_df[col] + 1)
        
        # Create education-based features
        if 'B15003' in col:  # Educational attainment
            # Calculate percentage of population with college degree
            if 'B15003_022E' in col or 'B15003_023E' in col:  # Bachelor's or higher
                total_edu_cols = [c for c in census_cols if 'B15003_001E' in c]
                if total_edu_cols:
                    total_edu = engineered_df[total_edu_cols[0]]
                    if total_edu.sum() > 0:
                        engineered_df[f"{col}_pct"] = engineered_df[col] / total_edu * 100
    
    # Create composite features
    # Income inequality (if we have income distribution data)
    income_cols = [col for col in census_cols if 'B19013' in col]
    if len(income_cols) >= 2:
        engineered_df['income_range'] = engineered_df[income_cols].max(axis=1) - engineered_df[income_cols].min(axis=1)
    
    # Population density proxy (if we have population data)
    pop_cols = [col for col in census_cols if 'B01001_001E' in col]
    if pop_cols:
        engineered_df['total_population'] = engineered_df[pop_cols[0]]
    
    logger.info(f"Engineered features added. New shape: {engineered_df.shape}")
    
    return engineered_df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between variables.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating interaction features...")
    
    interaction_df = df.copy()
    
    # Get demographic columns
    demographic_cols = [col for col in df.columns if re.match(r'^B\d+.*E$', col)]
    
    # Create meaningful interactions
    interactions_created = 0
    
    # Income × Education interactions
    income_cols = [col for col in demographic_cols if 'B19013' in col]
    education_cols = [col for col in demographic_cols if 'B15003' in col]
    
    if income_cols and education_cols:
        for income_col in income_cols[:2]:  # Limit to first 2 income variables
            for edu_col in education_cols[:2]:  # Limit to first 2 education variables
                interaction_name = f"{income_col}_x_{edu_col}"
                interaction_df[interaction_name] = interaction_df[income_col] * interaction_df[edu_col]
                interactions_created += 1
    
    # Population × Income interactions
    pop_cols = [col for col in demographic_cols if 'B01001_001E' in col]
    if pop_cols and income_cols:
        for pop_col in pop_cols[:1]:
            for income_col in income_cols[:1]:
                interaction_name = f"{pop_col}_x_{income_col}"
                interaction_df[interaction_name] = interaction_df[pop_col] * interaction_df[income_col]
                interactions_created += 1
    
    # Age × Income interactions (if age data available)
    age_cols = [col for col in demographic_cols if 'B01001' in col and any(age in col for age in ['_002E', '_026E'])]
    if age_cols and income_cols:
        for age_col in age_cols[:1]:
            for income_col in income_cols[:1]:
                interaction_name = f"{age_col}_x_{income_col}"
                interaction_df[interaction_name] = interaction_df[age_col] * interaction_df[income_col]
                interactions_created += 1
    
    logger.info(f"Created {interactions_created} interaction features")
    
    return interaction_df


def scale_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Scale features using StandardScaler.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature columns to scale
        
    Returns:
        DataFrame with scaled features
    """
    logger.info(f"Scaling {len(feature_cols)} features...")
    
    scaled_df = df.copy()
    
    # Filter to only include columns that exist in the dataframe
    existing_cols = [col for col in feature_cols if col in df.columns]
    
    if not existing_cols:
        logger.warning("No valid feature columns found for scaling")
        return scaled_df
    
    # Remove any columns with all NaN values
    valid_cols = []
    for col in existing_cols:
        if not df[col].isna().all():
            valid_cols.append(col)
    
    if not valid_cols:
        logger.warning("No valid features found for scaling (all columns are NaN)")
        return scaled_df
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale the features
    scaled_features = scaler.fit_transform(scaled_df[valid_cols].fillna(0))
    
    # Create new column names for scaled features
    scaled_col_names = [f"{col}_scaled" for col in valid_cols]
    
    # Add scaled features to dataframe
    for i, col_name in enumerate(scaled_col_names):
        scaled_df[col_name] = scaled_features[:, i]
    
    logger.info(f"Successfully scaled {len(valid_cols)} features")
    
    return scaled_df


def prepare_modeling_dataset(
    election_data: pd.DataFrame,
    demographics_data: pd.DataFrame,
    registered_voters_data: pd.DataFrame = None,
    target_variable: str = "democratic_share",
    feature_cols: Optional[List[str]] = None,
    scale_features_flag: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete pipeline to prepare dataset for modeling.
    
    Args:
        election_data: Raw election results
        demographics_data: Raw demographics data
        registered_voters_data: Optional registered voters data for turnout calculation
        target_variable: Target variable to predict
        feature_cols: List of feature columns to include (if None, uses all census variables)
        scale_features_flag: Whether to scale features
        
    Returns:
        Tuple of (processed DataFrame, list of feature column names)
    """
    logger.info("Preparing modeling dataset...")
    
    # Step 1: Join election and demographics data
    joined_df = join_election_demographics(election_data, demographics_data, registered_voters_data)
    
    if joined_df.empty:
        logger.error("Failed to join election and demographics data")
        return pd.DataFrame(), []
    
    # Step 2: Create lag features
    joined_df = create_lag_features(joined_df, target_variable)
    
    # Step 3: Engineer demographic features
    joined_df = engineer_demographic_features(joined_df)
    
    # Step 4: Create interaction features
    joined_df = create_interaction_features(joined_df)
    
    # Step 5: Identify feature columns
    if feature_cols is None:
        # Use all census variables and engineered features
        feature_cols = [col for col in joined_df.columns 
                       if re.match(r'^B\d+.*E', col) or 
                          col.endswith('_pct') or 
                          col.endswith('_log') or 
                          col.endswith('_scaled') or
                          col.startswith('B') and '_x_' in col]
    
    # Remove any feature columns that don't exist
    feature_cols = [col for col in feature_cols if col in joined_df.columns]
    
    # Step 6: Scale features if requested
    if scale_features_flag:
        joined_df = scale_features(joined_df, feature_cols)
        # Update feature columns to include scaled versions
        scaled_feature_cols = [col for col in joined_df.columns if col.endswith('_scaled')]
        feature_cols = [col for col in feature_cols if not col.endswith('_scaled')] + scaled_feature_cols
    
    # Remove rows with missing target variable
    final_df = joined_df.dropna(subset=[target_variable])
    
    logger.info(f"Final dataset shape: {final_df.shape}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Target variable: {target_variable}")
    
    return final_df, feature_cols


if __name__ == "__main__":
    # Example usage
    logger.info("Feature engineering module loaded successfully") 