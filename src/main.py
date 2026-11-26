"""
Master Execution Pipeline:
Orchestrates end-to-end authentic data loading, leak-free feature engineering,
Repeated Stratified K-Fold CV, hierarchical & cross-wave transfer benchmarks,
Platt calibration + DeLong/Wilcoxon tests, fairness & counterfactual audits,
TreeSHAP explainability + interaction discovery, and full econometric suite.
"""

import os
import sys
import warnings

from data_loader import load_findex_data
from econometrics import run_full_econometric_suite
from explainability import (
    generate_econometric_figures,
    generate_interactions_and_calibration_figure,
    run_shap_analysis,
)
from models import (
    compare_models,
    compute_calibration_curve_data,
    make_calibrated_stack,
    make_lgb_classifier,
    make_xgb_classifier,
    run_cross_wave_transfer,
    run_fairness_audit,
    run_hierarchical_ablation,
    run_policy_counterfactual_simulation,
    run_repeated_cv_benchmark,
    run_statistical_comparisons,
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

    cash_reliant_count = int((capable_df['cash_reliant'] == 1).sum())
    digital_user_count = int((capable_df['cash_reliant'] == 0).sum())
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

    # 4. Repeated Stratified K-Fold CV Ablation + Hierarchical Block Ablation
    print("\n[Step 4] Running 5x3 Repeated Stratified K-Fold CV Ablation (Survey-Weighted)...")
    cv_df, fold_aucs = run_repeated_cv_benchmark(
        data['X_full'],
        data['y_full'],
        data['w_full'],
        data['baseline_features'],
        random_state=42,
    )
    cv_df.to_csv('figures/cv_ablation_table.csv', index=False)

    print("\n[Step 4b] Running Hierarchical Domain-Block Ablation (Demographics -> Cash-Out -> Depth -> Full)...")
    hier_df = run_hierarchical_ablation(
        data['X_full'],
        data['y_full'],
        data['w_full'],
        data['feature_blocks'],
        random_state=42,
    )
    hier_df.to_csv('figures/hierarchical_ablation.csv', index=False)
    print(hier_df.to_string(index=False))

    print("\n[Step 4c] Running Cross-Wave Temporal Generalization (2021 <-> 2024)...")
    transfer_df = run_cross_wave_transfer(capable_df, data['feature_names'], random_state=42)
    transfer_df.to_csv('figures/cross_wave_transfer.csv', index=False)
    print(transfer_df.to_string(index=False))

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

    # 6. Evaluate and Compare Held-Out Models + Statistical Tests + Calibration + Fairness + Counterfactuals
    print("\n[Step 6] Held-Out Test Set Evaluation (Survey-Weighted):")
    results_map = {
        'Spline-Logit (Baseline+)': logit_res,
        'XGBoost (Reg+FE)': xgb_res,
        'LightGBM (Reg+FE)': lgb_res,
        'Stacked + Platt Cal': stack_res,
    }
    comp_df = compare_models(results_map)
    print(comp_df.to_string(index=False))
    comp_df.to_csv('figures/model_comparison_table.csv', index=False)
    logit_res['odds_ratios'].to_csv('figures/logit_odds_ratios.csv')

    stat_df = run_statistical_comparisons(
        fold_aucs,
        data['y_test'],
        {k: v['y_pred_prob'] for k, v in results_map.items()},
    )
    stat_df.to_csv('figures/statistical_tests.csv', index=False)

    cal_df = compute_calibration_curve_data(
        data['y_test'],
        {
            'Spline-Logit': logit_res['y_pred_prob'],
            'Regularized XGBoost': xgb_res['y_pred_prob'],
            'Stacked + Platt Cal': stack_res['y_pred_prob'],
        },
        data['w_test'],
    )
    cal_df.to_csv('figures/calibration_reliability.csv', index=False)

    fair_df = run_fairness_audit(data['X_test'], data['y_test'], data['w_test'], stack_res['y_pred_prob'])
    fair_df.to_csv('figures/fairness_audit.csv', index=False)

    cf_df = run_policy_counterfactual_simulation(stack_res['model'], capable_df, data['feature_names'])
    cf_df.to_csv('figures/policy_counterfactuals.csv', index=False)
    print("\n[Step 6b] Model-Implied Policy Counterfactuals (2024 Capable Cohort):")
    print(cf_df.to_string(index=False))

    # 7. TreeSHAP Explainability + Interaction Discovery
    print("\n[Step 7] Computing TreeSHAP Values, Interactions & Generating High-Res Visualizations...")
    shap_results = run_shap_analysis(
        xgb_res['model'],
        data['X_train'],
        data['X_test'],
        data['feature_names'],
        output_dir='figures',
    )
    generate_interactions_and_calibration_figure(output_dir='figures', comp_df=comp_df)

    print("\n[Step 8] Top Ranked Leak-Free Drivers (TreeSHAP):")
    print(shap_results['importance_df'].head(12).to_string(index=False))

    # 8. Econometric Suite
    print("\n[Step 9] Running Full Econometric & Behavioral Audit Suite...")
    run_full_econometric_suite(capable_df, output_dir='figures')
    generate_econometric_figures(capable_df, output_dir='figures')

    print("\n" + "=" * 80)
    print("LEAK-FREE 2-WAVE EMPIRICAL PIPELINE COMPLETED SUCCESSFULLY!")
    print("All CV tables, statistical tests, SHAP figures, and econometric CSVs exported to 'figures/'")
    print("=" * 80)


if __name__ == '__main__':
    main()
