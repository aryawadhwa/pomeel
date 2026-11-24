"""
Master Execution Pipeline:
Orchestrates end-to-end authentic data loading, leak-free feature engineering,
Repeated Stratified K-Fold CV, multi-family stacking + Platt calibration,
out-of-fold threshold optimization, and TreeSHAP explainability.
"""

import os
import sys
import warnings

from data_loader import load_findex_data
from econometrics import run_full_econometric_suite
from explainability import generate_econometric_figures, run_shap_analysis
from models import (
    compare_models,
    make_calibrated_stack,
    make_lgb_classifier,
    make_xgb_classifier,
    run_repeated_cv_benchmark,
    train_logistic_baseline,
    train_model_with_postprocessing,
)
from preprocessor import filter_capable_cohort, prepare_features_and_targets

warnings.filterwarnings('ignore')


def main():
    print("=" * 80)
    print("CAPABLE-BUT-CASH PARADOX: LEAK-FREE 2-WAVE EMPIRICAL & XAI PIPELINE")
    print("=" * 80)

    # 1. Load Data (Strictly authentic World Bank microdata)
    print("\n[Step 1] Loading official World Bank Global Findex (2021 & 2024) India microdata...")
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

    cash_reliant_count = (capable_df['cash_reliant'] == 1).sum()
    digital_user_count = (capable_df['cash_reliant'] == 0).sum()
    print(f"  - Cash-Reliant (Capable but cash-only in store): {cash_reliant_count:,} ({cash_reliant_count/n_capable*100:.2f}%)")
    print(f"  - Digital Adopters (Made digital merchant payment): {digital_user_count:,} ({digital_user_count/n_capable*100:.2f}%)")

    os.makedirs('data/processed', exist_ok=True)
    capable_df.to_csv('data/processed/capable_cohort.csv', index=False)
    print("Saved processed cohort to data/processed/capable_cohort.csv")

    # 3. Partition Data (Leak-Free Feature Set)
    print("\n[Step 3] Preparing strictly leak-free features & stratified train/test partitions...")
    data = prepare_features_and_targets(capable_df, test_size=0.20, random_state=42)
    print(f"Train samples: {len(data['X_train'])}, Test samples: {len(data['X_test'])}")
    print(f"Leak-free feature dimensionality: {len(data['feature_names'])} features (vs {len(data['baseline_features'])} baseline)")

    os.makedirs('figures', exist_ok=True)

    # 4. Repeated Stratified K-Fold CV Ablation (15 evaluations, survey-weighted)
    print("\n[Step 4] Running 5x3 Repeated Stratified K-Fold CV Ablation (Survey-Weighted)...")
    cv_df = run_repeated_cv_benchmark(
        data['X_full'],
        data['y_full'],
        data['w_full'],
        data['baseline_features'],
        random_state=42,
    )
    cv_df.to_csv('figures/cv_ablation_table.csv', index=False)

    # 5. Train Held-Out Models with OOF Threshold Tuning & Calibration
    print("\n[Step 5] Fitting Held-Out Models (Spline-Logit, XGBoost, LightGBM, Calibrated Stack)...")
    logit_res = train_logistic_baseline(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test'],
    )
    xgb_res = train_model_with_postprocessing(
        lambda: make_xgb_classifier(42),
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test'],
    )
    lgb_res = train_model_with_postprocessing(
        lambda: make_lgb_classifier(42),
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test'],
    )
    stack_res = train_model_with_postprocessing(
        lambda: make_calibrated_stack(42),
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test'],
        tune_thresholds=False,
    )

    # 6. Evaluate and Compare Held-Out Models
    print("\n[Step 6] Held-Out Test Set Evaluation (Survey-Weighted):")
    comp_df = compare_models(
        {
            'Spline-Logit (Baseline+)': logit_res,
            'XGBoost (Reg+FE)': xgb_res,
            'LightGBM (Reg+FE)': lgb_res,
            'Stacked + Platt Cal': stack_res,
        }
    )
    print(comp_df.to_string(index=False))
    comp_df.to_csv('figures/model_comparison_table.csv', index=False)

    # Save GLM Odds Ratios
    logit_res['odds_ratios'].to_csv('figures/logit_odds_ratios.csv')

    # 7. TreeSHAP Explainability (on Regularized Leak-Free XGBoost)
    print("\n[Step 7] Computing TreeSHAP Values & Generating High-Res Visualizations...")
    shap_results = run_shap_analysis(
        xgb_res['model'],
        data['X_train'],
        data['X_test'],
        data['feature_names'],
        output_dir='figures',
    )

    print("\n[Step 8] Top Ranked Leak-Free Drivers (TreeSHAP):")
    print(shap_results['importance_df'].head(12).to_string(index=False))

    # 8. Econometric Suite (AMEs, Fairlie Decomposition, Heterogeneity, 2024 Tracker)
    print("\n[Step 9] Running Full Econometric & Behavioral Audit Suite...")
    run_full_econometric_suite(capable_df, output_dir='figures')
    generate_econometric_figures(capable_df, output_dir='figures')

    print("\n" + "=" * 80)
    print("LEAK-FREE 2-WAVE EMPIRICAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("All CV tables, model benchmarks, and SHAP figures exported to 'figures/'")
    print("=" * 80)


if __name__ == '__main__':
    main()
