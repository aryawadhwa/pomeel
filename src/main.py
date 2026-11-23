"""
Master Execution Pipeline:
Orchestrates end-to-end data loading, preprocessing, modeling,
TreeSHAP explainability, and exports publication artifacts.
STRICT PROTOCOL: Requires authentic World Bank microdata. No dummy data.
"""

import sys
import os
import pandas as pd
from data_loader import load_findex_data
from preprocessor import filter_capable_cohort, prepare_features_and_targets
from models import train_logistic_baseline, train_xgboost_classifier, compare_models
from explainability import run_shap_analysis


def main():
    print("=" * 75)
    print("CAPABLE-BUT-CASH PARADOX: AUTHENTIC EMPIRICAL RESEARCH PIPELINE")
    print("=" * 75)
    
    # 1. Load Data (Strictly authentic World Bank microdata)
    print("\n[Step 1] Loading official World Bank Global Findex 2021 India microdata...")
    try:
        raw_df = load_findex_data()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
        
    print(f"Total surveyed respondents in official dataset: {len(raw_df):,}")
    
    # 2. Filter Cohort
    print("\n[Step 2] Isolating the 'Capable' cohort (Account=1, Mobile=1, Internet=1)...")
    capable_df = filter_capable_cohort(raw_df)
    n_capable = len(capable_df)
    pct_capable = (n_capable / len(raw_df)) * 100
    print(f"Digitally capable sub-population size: {n_capable:,} ({pct_capable:.2f}% of total surveyed)")
    
    # Target breakdown
    cash_reliant_count = (capable_df['cash_reliant'] == 1).sum()
    digital_user_count = (capable_df['cash_reliant'] == 0).sum()
    print(f"  - Cash-Reliant (Capable but cash-only in store): {cash_reliant_count:,} ({cash_reliant_count/n_capable*100:.2f}%)")
    print(f"  - Digital Adopters (Made digital merchant payment): {digital_user_count:,} ({digital_user_count/n_capable*100:.2f}%)")
    
    # Save processed cohort
    os.makedirs('data/processed', exist_ok=True)
    capable_df.to_csv('data/processed/capable_cohort.csv', index=False)
    print("Saved processed cohort to data/processed/capable_cohort.csv")
    
    # 3. Partition Data
    print("\n[Step 3] Preparing feature matrix and stratified train/test partitions...")
    data = prepare_features_and_targets(capable_df, test_size=0.20, random_state=42)
    print(f"Train samples: {len(data['X_train'])}, Test samples: {len(data['X_test'])}")
    print(f"Feature space dimensionality: {len(data['feature_names'])} features")
    
    # 4. Train Models
    print("\n[Step 4] Fitting Survey-Weighted Logistic Regression Baseline...")
    logit_res = train_logistic_baseline(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test']
    )
    
    print("\n[Step 5] Training Regularized XGBoost Classifier...")
    xgb_res = train_xgboost_classifier(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test'],
        random_state=42
    )
    
    # 5. Evaluate and Compare
    print("\n[Step 6] Comparative Model Performance Evaluation:")
    comp_df = compare_models(logit_res, xgb_res)
    print(comp_df.to_string(index=False))
    os.makedirs('figures', exist_ok=True)
    comp_df.to_csv('figures/model_comparison_table.csv', index=False)
    
    # 6. TreeSHAP Explainability
    print("\n[Step 7] Computing TreeSHAP Values & Generating High-Res Visualizations...")
    shap_results = run_shap_analysis(
        xgb_res['model'],
        data['X_train'],
        data['X_test'],
        data['feature_names'],
        output_dir='figures'
    )
    
    print("\n[Step 8] Top Ranked Behavioral & Demographic Barriers (TreeSHAP):")
    print(shap_results['importance_df'].to_string(index=False))
    
    print("\n" + "=" * 75)
    print("AUTHENTIC EMPIRICAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("All tables and figures exported to 'figures/' and 'data/processed/'")
    print("=" * 75)


if __name__ == '__main__':
    main()
