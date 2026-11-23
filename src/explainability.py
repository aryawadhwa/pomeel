"""
Explainability Module using TreeSHAP:
Computes global feature attributions, directional beeswarm plots,
interaction/dependence curves, and local waterfall explanations.
Exports high-resolution figures for the research paper.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(
    xgb_model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
    output_dir: str = 'figures'
) -> Dict[str, Any]:
    """
    Executes complete TreeSHAP interpretability pipeline and generates publication-grade plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'font.size': 11})
    
    # Initialize TreeSHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)
    
    # 1. Global Feature Importance (Mean |SHAP|)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)
    
    importance_df.to_csv(os.path.join(output_dir, 'shap_barriers_table.csv'), index=False)
    
    plt.figure(figsize=(9, 6))
    plt.barh(importance_df['Feature'][::-1], importance_df['Mean_Absolute_SHAP'][::-1], color='#1f77b4', edgecolor='black')
    plt.xlabel('Mean |SHAP Value| (Impact on Cash Reliance Prediction)', fontsize=12, fontweight='bold')
    plt.title('Global Feature Importance Ranking (TreeSHAP)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_global_importance.png'), dpi=300)
    plt.close()
    
    # 2. Directional Summary (Beeswarm) Plot
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title('SHAP Directional Summary (Impact on Predicting Cash Reliance)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300)
    plt.close()
    
    # 3. Dependence Plot: Age vs SHAP Value (colored by Education or Wage Cash)
    plt.figure(figsize=(8, 6))
    shap.plots.scatter(shap_values[:, "age"], color=shap_values[:, "wage_cash"], show=False)
    plt.title('SHAP Dependence: Age Non-Linearity & Wage Cash Interaction', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_dependence_age.png'), dpi=300)
    plt.close()
    
    # 4. Local Explanations: Representative Personas
    # Persona A: High probability of cash reliance (Cash-reliant archetype)
    pred_probs = xgb_model.predict_proba(X_test)[:, 1]
    cash_idx = np.argmax(pred_probs)
    digital_idx = np.argmin(pred_probs)
    
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[cash_idx], max_display=10, show=False)
    plt.title(f'Case Study A: Capable Cash-Reliant Persona (P(Cash)={pred_probs[cash_idx]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_waterfall_cash_case.png'), dpi=300)
    plt.close()
    
    # Persona B: High probability of digital adoption (Digital adopter archetype)
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[digital_idx], max_display=10, show=False)
    plt.title(f'Case Study B: Digital Adopter Persona (P(Cash)={pred_probs[digital_idx]:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_waterfall_digital_case.png'), dpi=300)
    plt.close()
    
    print(f"All SHAP interpretability figures successfully saved to '{output_dir}/'")
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'importance_df': importance_df
    }


if __name__ == '__main__':
    from data_loader import load_findex_data
    from preprocessor import filter_capable_cohort, prepare_features_and_targets
    from models import train_xgboost_classifier
    
    raw_df = load_findex_data()
    capable_df = filter_capable_cohort(raw_df)
    data = prepare_features_and_targets(capable_df)
    
    xgb_res = train_xgboost_classifier(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test']
    )
    
    shap_results = run_shap_analysis(
        xgb_res['model'],
        data['X_train'],
        data['X_test'],
        data['feature_names'],
        output_dir='figures'
    )
    
    print("\nTop 5 Ranked Barriers to Digital Merchant Adoption:")
    print(shap_results['importance_df'].head(5).to_string(index=False))
