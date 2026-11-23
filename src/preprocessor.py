"""
Preprocessor module for isolating the 'Capable-but-Cash' cohort,
feature engineering, survey weight handling, and stratified train/test partitioning.
"""

from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def filter_capable_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolates the 'Capable' segment: respondents possessing an active bank account,
    mobile phone ownership, and internet access.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full surveyed sample.
        
    Returns
    -------
    pd.DataFrame
        Filtered sub-population of digitally capable adults.
    """
    condition = (df['account'] == 1) & (df['mobileowner'] == 1) & (df['internet'] == 1)
    capable_df = df[condition].copy()
    
    # Formulate binary target:
    # 1 = Cash-Reliant (Capable but relies exclusively on cash in-store)
    # 0 = Digital User (Made a digital merchant payment)
    capable_df['cash_reliant'] = (capable_df['merchantpay_dig'] == 0).astype(int)
    
    return capable_df


def prepare_features_and_targets(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Extracts predictor feature matrix, target vector, and survey sampling weights.
    Performs stratified train/test split.
    
    Parameters
    ----------
    df : pd.DataFrame
        Capable cohort dataframe.
    test_size : float
        Fraction of cohort reserved for held-out evaluation.
    random_state : int
        Seed for reproducibility.
        
    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test, w_train, w_test, and feature_names.
    """
    feature_cols = [
        'age',
        'female',
        'educ',
        'inc_q',
        'urbanicity',
        'fin2',
        'fin3',
        'wage_cash',
        'gov_cash',
        'withdrew_atm',
        'emergency_cost',
        'barrier_complexity',
        'barrier_trust',
        'barrier_cost',
        'barrier_dist'
    ]
    
    # Verify presence of all columns
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' missing from dataframe.")
            
    X = df[feature_cols].copy()
    y = df['cash_reliant'].values
    weights = df['wgt'].values if 'wgt' in df.columns else np.ones(len(df))
    
    # Stratified split preserving class balance and weights
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    w_train = weights[train_idx]
    w_test = weights[test_idx]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'w_train': w_train,
        'w_test': w_test,
        'feature_names': feature_cols,
        'full_capable_df': df
    }


if __name__ == '__main__':
    from data_loader import load_findex_data
    
    raw_df = load_findex_data()
    capable_df = filter_capable_cohort(raw_df)
    print(f"Total surveyed: {len(raw_df)}")
    print(f"Capable segment: {len(capable_df)} ({len(capable_df)/len(raw_df)*100:.2f}%)")
    print(f"Target distribution (1 = Cash-Reliant, 0 = Digital User):")
    print(capable_df['cash_reliant'].value_counts(normalize=True))
    
    data = prepare_features_and_targets(capable_df)
    print(f"\nTrain set size: {len(data['X_train'])}, Test set size: {len(data['X_test'])}")
    print("Features:", data['feature_names'])
