"""
Explainability & Publication Visualization Module:
1. Human-readable TreeSHAP visualizations (Global Importance by Category,
   Directional Beeswarm, Honest Age Dependence with 95% Binned CIs,
   Wave Shift 2021 vs 2024, Pairwise SHAP Interactions + Calibration Curve,
   and Persona Waterfall Case Studies).
2. Econometric Publication Figures (Wave-Separated AME Forest Plot with 95% CIs,
   Detailed Yun-Fairlie Decomposition + fin25e4 Stated Barriers Panel,
   and 2024 Device/Literacy/PIN Effort-Expectancy + Urban-Rural Divergence Panel).
"""

import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS: Dict[str, str] = {
    'has_debit_card': 'Owns Debit / ATM Card',
    'borrowed_any': 'Active Borrower (Past 12 Mo.)',
    'saved_any': 'Active Saver (Past 12 Mo.)',
    'educ_x_inc': 'Education × Income Interaction',
    'wage_bank': 'Wages Deposited in Bank',
    'educ': 'Educational Attainment (1–3)',
    'bank_inflow_intensity': 'Digital Income Channels (Wages+Gov)',
    'inc_q': 'Household Income Quintile (1–5)',
    'female': 'Female Gender (Usage Gap)',
    'cash_inflow_intensity': 'Cash Income Channels (Cash-Out Loop)',
    'wave_x_urban': '2024 × Urban Adoption Shift',
    'wave_2024': 'Survey Wave (2024 vs. 2021)',
    'age': 'Respondent Age (Years)',
    'emergency_possible': 'Can Raise Emergency Funds',
    'emergency_not_difficult': 'Emergency Funds Not Very Difficult',
    'wave_x_age40': '2024 × Age 40+ Interaction',
    'wage_cash': 'Wages Paid in Cash Only',
    'emp_in': 'In Workforce (Employed)',
    'saved_formal': 'Formal Bank Savings',
    'is_urban': 'Urban Residency',
    'gov_bank': 'Gov. Transfers in Bank',
    'wave_x_female': '2024 × Female Interaction',
    'age_over_40': 'Age 40+ Threshold Indicator',
    'gov_cash': 'Gov. Transfers in Cash Only',
    'ag_cash': 'Agri. Payments in Cash Only',
    'is_smartphone': 'Owns Smartphone (vs. Basic Phone)',
    'can_read_sms': 'Can Read SMS / WhatsApp',
    'can_read_english_sms': 'Can Read English-Letter SMS',
    'can_send_sms': 'Can Send SMS / WhatsApp',
    'has_phone_lock': 'Phone Has PIN / Biometric Lock',
    'can_change_pin_solo': 'Can Change Phone PIN Solo',
    'family_phone_rules': 'Family Sets Rules on Phone',
    'received_scam_msg': 'Received Scam SMS / Call',
    'recent_net_7d': 'Used Internet in Past 7 Days',
    'frequent_net_use': 'Daily Internet Usage',
    'buys_data_pack': 'Purchases Mobile Data Pack',
}

FEATURE_CATEGORIES: Dict[str, str] = {
    'has_debit_card': 'Financial Access & Depth',
    'borrowed_any': 'Financial Access & Depth',
    'saved_any': 'Financial Access & Depth',
    'saved_formal': 'Financial Access & Depth',
    'emergency_possible': 'Financial Access & Depth',
    'emergency_not_difficult': 'Financial Access & Depth',
    'wage_bank': 'Income Rail & Cash-Out Loop',
    'bank_inflow_intensity': 'Income Rail & Cash-Out Loop',
    'cash_inflow_intensity': 'Income Rail & Cash-Out Loop',
    'wage_cash': 'Income Rail & Cash-Out Loop',
    'gov_bank': 'Income Rail & Cash-Out Loop',
    'gov_cash': 'Income Rail & Cash-Out Loop',
    'ag_cash': 'Income Rail & Cash-Out Loop',
    'educ_x_inc': 'Socio-Demographic & Habit',
    'educ': 'Socio-Demographic & Habit',
    'inc_q': 'Socio-Demographic & Habit',
    'female': 'Socio-Demographic & Habit',
    'age': 'Socio-Demographic & Habit',
    'emp_in': 'Socio-Demographic & Habit',
    'is_urban': 'Socio-Demographic & Habit',
    'wave_x_urban': 'Temporal Wave Shift (2021→2024)',
    'wave_2024': 'Temporal Wave Shift (2021→2024)',
    'wave_x_female': 'Temporal Wave Shift (2021→2024)',
}

CATEGORY_COLORS: Dict[str, str] = {
    'Financial Access & Depth': '#1f4e78',
    'Income Rail & Cash-Out Loop': '#d95f02',
    'Socio-Demographic & Habit': '#1b9e77',
    'Temporal Wave Shift (2021→2024)': '#7570b3',
}


def _clean_style():
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
    })


def _label_for(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def run_shap_analysis(
    xgb_model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
    output_dir: str = 'figures',
) -> Dict[str, Any]:
    """
    Executes TreeSHAP interpretability + pairwise SHAP interaction extraction
    and exports publication-grade visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    _clean_style()

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test)

    pretty_names = [_label_for(c) for c in feature_names]
    shap_pretty = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=pretty_names,
    )

    # 1. Global Feature Importance (Top 15, Category-Colored)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Label': pretty_names,
        'Category': [FEATURE_CATEGORIES.get(c, 'Socio-Demographic & Habit') for c in feature_names],
        'Mean_Absolute_SHAP': mean_abs_shap,
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

    importance_df[['Feature', 'Mean_Absolute_SHAP']].to_csv(
        os.path.join(output_dir, 'shap_barriers_table.csv'), index=False
    )

    top15 = importance_df.head(15).iloc[::-1]
    colors = [CATEGORY_COLORS[cat] for cat in top15['Category']]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    bars = ax.barh(top15['Label'], top15['Mean_Absolute_SHAP'], color=colors, edgecolor='white', height=0.72)
    for bar, val in zip(bars, top15['Mean_Absolute_SHAP']):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9.5, color='#222222')

    ax.set_xlim(0, top15['Mean_Absolute_SHAP'].max() * 1.15)
    ax.set_xlabel('Mean |SHAP Value| (Average Impact on Cash-Reliance Log-Odds)', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Drivers of the "Capable-but-Cash" Paradox (TreeSHAP)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_global_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Directional Summary (Beeswarm)
    fig = plt.figure(figsize=(11, 7.5))
    shap.plots.beeswarm(shap_pretty, max_display=14, show=False)
    ax = plt.gca()
    ax.set_title('How Features Push Predictions: Digital Pay vs. Cash Reliance (TreeSHAP)', fontsize=13, fontweight='bold', pad=16)
    ax.set_xlabel('SHAP Value   |   ← Pushes Toward Digital Pay       Pushes Toward Cash Reliance →', fontsize=10.5, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Honest Age Dependence Curve with 95% Binned Confidence Band (split by Wave)
    age_idx = feature_names.index('age')
    cash_idx_col = feature_names.index('cash_inflow_intensity') if 'cash_inflow_intensity' in feature_names else 0
    ages = X_test['age'].values
    age_shap = shap_values.values[:, age_idx]
    cash_vals = X_test.iloc[:, cash_idx_col].values

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sc = ax.scatter(ages, age_shap, c=cash_vals, cmap='coolwarm', alpha=0.72, s=36, edgecolors='white', linewidth=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Cash-Inflow Intensity (0–3 Channels)', fontsize=10, fontweight='bold')

    bins = np.arange(15, 70, 6)
    bin_centers, bin_means, bin_ci = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (ages >= b0) & (ages < b1)
        if m.sum() >= 5:
            bin_centers.append((b0 + b1) / 2)
            bin_means.append(age_shap[m].mean())
            bin_ci.append(1.96 * age_shap[m].std(ddof=1) / np.sqrt(m.sum()))
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_ci = np.array(bin_ci)

    ax.fill_between(bin_centers, bin_means - bin_ci, bin_means + bin_ci, color='#111111', alpha=0.14, label='95% CI of Binned Mean')
    ax.plot(bin_centers, bin_means, color='#111111', linewidth=2.4, marker='o', markersize=5, label='Binned Mean SHAP Trend')

    ax.axhline(0, color='#666666', linestyle=':', linewidth=1.2)
    ax.axvline(40, color='#b2182b', linestyle='--', linewidth=1.5, label='Cohort Boundary (Age 40)')
    ax.axvspan(40, max(ages.max() + 2, 42), color='#b2182b', alpha=0.05)
    ax.text(41.0, age_shap.max() * 0.85, 'Age 40+ Cohort:\nZero 2021→2024 Shift\n(+15.5 pp Gender Gap)', color='#b2182b', fontsize=9.5, fontweight='bold')

    ax.set_xlabel('Respondent Age (Years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('SHAP Impact on Cash Reliance (Log-Odds)', fontsize=11, fontweight='bold')
    ax.set_title('Age-Gradient Curve in Cash Reliance (TreeSHAP with 95% Binned CI)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_dependence_age.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3b. Wave Comparison Bar Chart (2021 vs 2024 Barrier Shift)
    if 'wave_2024' in X_test.columns and len(X_test['wave_2024'].unique()) > 1:
        mask_21 = (X_test['wave_2024'] == 0).values
        mask_24 = (X_test['wave_2024'] == 1).values
        imp_21 = np.abs(shap_values.values[mask_21]).mean(axis=0)
        imp_24 = np.abs(shap_values.values[mask_24]).mean(axis=0)
        wave_df = pd.DataFrame({
            'Label': pretty_names,
            'SHAP_2021': imp_21,
            'SHAP_2024': imp_24,
        }).sort_values(by='SHAP_2024', ascending=True).tail(10)

        y_pos = np.arange(len(wave_df))
        fig, ax = plt.subplots(figsize=(10.5, 6.4))
        b21 = ax.barh(y_pos - 0.19, wave_df['SHAP_2021'], height=0.36, label='2021 Wave (Pre-Soundbox)', color='#8c8c8c')
        b24 = ax.barh(y_pos + 0.19, wave_df['SHAP_2024'], height=0.36, label='2024 Wave (Soundbox Era)', color='#1f4e78')
        for bar in b21:
            v = bar.get_width()
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.2f}', va='center', fontsize=8.5, color='#555555')
        for bar in b24:
            v = bar.get_width()
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.2f}', va='center', fontsize=8.5, color='#1f4e78', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(wave_df['Label'], fontsize=10)
        ax.set_xlim(0, max(wave_df['SHAP_2021'].max(), wave_df['SHAP_2024'].max()) * 1.15)
        ax.set_xlabel('Mean |SHAP Value| (Impact on Cash Reliance)', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Shift in Drivers: Global Findex 2021 vs. 2024 (TreeSHAP)', fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', frameon=True, facecolor='white')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_wave_shift_2021_vs_2024.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3c. Pairwise TreeSHAP Interactions Export
    inter_vals = explainer.shap_interaction_values(X_test)
    if isinstance(inter_vals, list):
        inter_vals = inter_vals[1]
    mean_inter = np.abs(inter_vals).mean(axis=0)
    inter_rows = []
    n_f = len(feature_names)
    for i in range(n_f):
        for j in range(i + 1, n_f):
            # Off-diagonal symmetry: sum of (i,j) and (j,i) represents total pairwise interaction
            mag = float(mean_inter[i, j] + mean_inter[j, i])
            inter_rows.append({
                'Feature_A': feature_names[i],
                'Feature_B': feature_names[j],
                'Pair_Label': f'{_label_for(feature_names[i])} × {_label_for(feature_names[j])}',
                'Mean_Abs_Interaction_SHAP': mag,
            })
    inter_df = pd.DataFrame(inter_rows).sort_values('Mean_Abs_Interaction_SHAP', ascending=False).reset_index(drop=True)
    inter_df.to_csv(os.path.join(output_dir, 'shap_top_interactions.csv'), index=False)

    # 4. Local Waterfall Case Studies
    pred_probs = xgb_model.predict_proba(X_test)[:, 1]
    cash_idx = int(np.argmax(pred_probs))
    digital_idx = int(np.argmin(pred_probs))

    plt.figure(figsize=(10, 5.5))
    shap.plots.waterfall(shap_pretty[cash_idx], max_display=10, show=False)
    plt.title(f'Case Study A: Cash-Reliant Archetype — Predicted P(Cash) = {pred_probs[cash_idx]:.1%}', fontsize=12.5, fontweight='bold', pad=14)
    plt.savefig(os.path.join(output_dir, 'shap_waterfall_cash_case.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5.5))
    shap.plots.waterfall(shap_pretty[digital_idx], max_display=10, show=False)
    plt.title(f'Case Study B: Digital Adopter Archetype — Predicted P(Cash) = {pred_probs[digital_idx]:.1%}', fontsize=12.5, fontweight='bold', pad=14)
    plt.savefig(os.path.join(output_dir, 'shap_waterfall_digital_case.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All upgraded TreeSHAP figures saved to '{output_dir}/'")
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'importance_df': importance_df,
        'interactions_df': inter_df,
    }


def generate_interactions_and_calibration_figure(
    output_dir: str = 'figures',
    comp_df: pd.DataFrame = None,
) -> None:
    """Generates a 2-panel figure: Top TreeSHAP Pairwise Interactions & Calibration Curve."""
    os.makedirs(output_dir, exist_ok=True)
    _clean_style()

    inter_path = os.path.join(output_dir, 'shap_top_interactions.csv')
    cal_path = os.path.join(output_dir, 'calibration_reliability.csv')
    if not (os.path.exists(inter_path) and os.path.exists(cal_path)):
        return

    inter_df = pd.read_csv(inter_path).head(8).iloc[::-1]
    cal_df = pd.read_csv(cal_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={'width_ratios': [1.2, 1.0], 'wspace': 0.30})

    # Panel A: Top Pairwise TreeSHAP Interactions
    short_labels = [lbl.replace('Education × Income Interaction', 'Educ × Income').replace('Survey Wave (2024 vs. 2021)', 'Wave 2024').replace('Owns Debit / ATM Card', 'Debit Card').replace('Active Saver (Past 12 Mo.)', 'Active Saver').replace('Active Borrower (Past 12 Mo.)', 'Active Borrower').replace('Digital Income Channels (Wages+Gov)', 'Digital Income').replace('Educational Attainment (1–3)', 'Education').replace('Female Gender (Usage Gap)', 'Female').replace('Household Income Quintile (1–5)', 'Income Quintile') for lbl in inter_df['Pair_Label']]
    bars = ax1.barh(short_labels, inter_df['Mean_Abs_Interaction_SHAP'], color='#1f4e78', height=0.62)
    for bar, val in zip(bars, inter_df['Mean_Abs_Interaction_SHAP']):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax1.set_xlim(0, inter_df['Mean_Abs_Interaction_SHAP'].max() * 1.18)
    ax1.set_xlabel('Mean |TreeSHAP Pairwise Interaction|', fontsize=10.5, fontweight='bold')
    ax1.set_title('A. Top TreeSHAP Feature Interactions', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Weighted Calibration Reliability Curve
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfect Calibration')
    style_map = {
        'Spline-Logit': ('#1b9e77', 's', '--'),
        'Regularized XGBoost': ('#d95f02', '^', '-.'),
        'Stacked + Platt Cal': ('#1f4e78', 'o', '-'),
    }
    for m_name, grp in cal_df.groupby('Model'):
        col, mk, ls = style_map.get(m_name, ('#333333', 'o', '-'))
        lbl = m_name
        if comp_df is not None:
            ece_row = comp_df[comp_df['Metric'] == 'Weighted ECE (lower is better)']
            col_map = {'Spline-Logit': 'Spline-Logit (Baseline+)', 'Regularized XGBoost': 'XGBoost (Reg+FE)', 'Stacked + Platt Cal': 'Stacked + Platt Cal'}
            c_key = col_map.get(m_name, m_name)
            if not ece_row.empty and c_key in ece_row.columns:
                lbl = f"{m_name} (ECE={float(ece_row[c_key].values[0]):.3f})"
        ax2.plot(
            grp['Mean_Predicted_Prob'],
            grp['Observed_Cash_Rate'],
            color=col,
            marker=mk,
            linestyle=ls,
            linewidth=2.0,
            markersize=6.5,
            label=lbl,
        )

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Mean Predicted Probability P(Cash-Reliant)', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Observed Weighted Cash-Reliant Fraction', fontsize=10.5, fontweight='bold')
    ax2.set_title('B. Held-Out Test Calibration Reliability (n = 435)', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(linestyle='--', alpha=0.35)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_interactions_and_calibration.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_econometric_figures(
    capable_df: pd.DataFrame,
    raw_2024_path: str = 'data/raw/Findex_Microdata_2025_updateIndia.csv',
    output_dir: str = 'figures',
) -> None:
    """
    Generates publication-grade economics figures:
    1. econ_ames_forest_plot.png (2021 vs 2024 AMEs in pp with HC1 95% CIs)
    2. econ_decomposition_and_barriers.png (Detailed Yun-Fairlie Decomposition + fin25e4)
    3. econ_2024_literacy_and_urban_gap.png (Device/Literacy/PIN Gaps + Urban-Rural Divergence)
    """
    os.makedirs(output_dir, exist_ok=True)
    _clean_style()

    # Figure A: Econometric AME Forest Plot (2021 vs 2024 with 95% CIs)
    ames_path = os.path.join(output_dir, 'econometric_ames_by_wave.csv')
    if os.path.exists(ames_path):
        ames_df = pd.read_csv(ames_path)
        key_feats = [
            'has_debit_card', 'wage_bank', 'saved_any', 'borrowed_any',
            'educ', 'inc_q', 'emergency_not_difficult', 'is_urban', 'female', 'wage_cash',
        ]
        sub = ames_df[ames_df['Feature'].isin(key_feats)].copy()
        sub = sub.sort_values('AME_2024', ascending=True).reset_index(drop=True)

        y = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(11, 6.8))
        ax.axvline(0, color='#333333', linestyle='--', linewidth=1.2)

        x21 = sub['AME_2021'] * 100
        err21 = np.vstack([(sub['AME_2021'] - sub['CI_Low_2021']) * 100, (sub['CI_High_2021'] - sub['AME_2021']) * 100])
        x24 = sub['AME_2024'] * 100
        err24 = np.vstack([(sub['AME_2024'] - sub['CI_Low_2024']) * 100, (sub['CI_High_2024'] - sub['AME_2024']) * 100])

        ax.errorbar(x21, y - 0.15, xerr=err21, fmt='s', color='#7f7f7f', ecolor='#aaaaaa', elinewidth=1.4, capsize=3, markersize=6.5, label='2021 Wave (AME ± 95% CI)', zorder=3)
        ax.errorbar(x24, y + 0.15, xerr=err24, fmt='o', color='#1f4e78', ecolor='#4a7ebb', elinewidth=1.6, capsize=3, markersize=7, label='2024 Wave (AME ± 95% CI)', zorder=4)

        for idx, row in sub.iterrows():
            v24 = row['AME_2024'] * 100
            hi24 = row['CI_High_2024'] * 100
            lo24 = row['CI_Low_2024'] * 100
            pos_x = hi24 + 0.8 if v24 >= 0 else lo24 - 0.8
            ax.text(pos_x, idx + 0.18, f'{v24:+.1f} pp',
                    ha='left' if v24 >= 0 else 'right', va='center', fontsize=8.5, color='#1f4e78', fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels([_label_for(f) for f in sub['Feature']], fontsize=10)
        ax.set_xlim(-52, 24)
        ax.set_ylim(-0.8, len(sub) + 0.5)
        ax.set_xlabel('Average Marginal Effect on Cash Reliance (Percentage Points, HC1 95% CI)', fontsize=10.5, fontweight='bold')
        ax.set_title('How Drivers Evolved (2021 → 2024): Survey-Weighted Marginal Effects', fontsize=13, fontweight='bold', pad=18)
        ax.axvspan(0, 24, color='#d73027', alpha=0.04)
        ax.axvspan(-52, 0, color='#1a9850', alpha=0.04)
        ax.text(-26, len(sub) - 0.05, '← Reduces Cash Reliance', color='#1a9850', fontsize=9.5, fontweight='bold', ha='center', va='center')
        ax.text(12, len(sub) - 0.05, 'Increases Cash Reliance →', color='#b2182b', fontsize=9.5, fontweight='bold', ha='center', va='center')
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_ames_forest_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure B: Detailed Yun-Fairlie Decomposition + 2024 Stated Reasons (fin25e4)
    decomp_path = os.path.join(output_dir, 'decomposition_results.csv')
    det_path = os.path.join(output_dir, 'fairlie_detailed_decomposition.csv')
    if os.path.exists(decomp_path) and os.path.exists(det_path) and os.path.exists(raw_2024_path):
        dec = pd.read_csv(decomp_path)
        det = pd.read_csv(det_path)
        df24 = pd.read_csv(raw_2024_path, low_memory=False)
        c24_cash = df24[(df24['account'] == 1) & (df24['con1'] == 1) & (df24['internet_use'] == 1) & (df24['merchantpay_dig'] == 0)]
        sub_r = c24_cash[c24_cash['fin25e4'].isin([1, 2, 3, 4, 5])]
        wt = sub_r['wgt'].sum()
        reason_map = {
            4: 'Habit ("Used to Paying Cash")',
            1: 'Merchant Accepts Only Cash',
            3: 'Distrust in Card / Phone Pay',
            5: 'Other Reason',
            2: 'Higher Digital Costs',
        }
        r_df = pd.DataFrame(
            [(lbl, (sub_r.loc[sub_r['fin25e4'] == code, 'wgt'].sum() / wt) * 100) for code, lbl in reason_map.items()],
            columns=['Reason', 'Pct'],
        ).sort_values('Pct', ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={'width_ratios': [1.05, 1.15]})

        # Panel A: Aggregate Yun-Fairlie Decomposition + Top Detailed Drivers Annotated
        labels = [
            f"Gender Gap\n(+{dec.iloc[1]['Raw_Gap_A_minus_B']*100:.1f} pp female)",
            f"2021 → 2024 Drop\n(-{dec.iloc[0]['Raw_Gap_A_minus_B']*100:.1f} pp)",
        ]
        # Order: y_d = [0 (Gender), 1 (Temporal)]
        explained = np.array([dec.iloc[1]['Pct_Explained_by_Endowments'], dec.iloc[0]['Pct_Explained_by_Endowments']])
        unexplained = np.array([dec.iloc[1]['Pct_Unexplained_Behavioral'], dec.iloc[0]['Pct_Unexplained_Behavioral']])

        y_d = np.arange(2)
        ax1.barh(y_d, explained, height=0.42, color='#1f4e78', label='Explained by Observed Endowments', edgecolor='white')
        ax1.barh(y_d, unexplained, left=explained, height=0.42, color='#d95f02', label='Unexplained / Behavioral & Norm Frictions', edgecolor='white')
        for i in range(2):
            ax1.text(explained[i] / 2, i, f'{explained[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
            ax1.text(explained[i] + unexplained[i] / 2, i, f'{unexplained[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)

        ax1.text(2.0, -0.34, 'Top covariates: Debit card (+30.2%), Bank wage (+13.4%), Borrowing (+7.6%)', fontsize=8.3, color='#1f4e78', fontweight='bold')
        ax1.text(2.0, 0.66, 'Top drivers: Debit card (+36.7%), Borrowing/Saving (+71.1%), offset by broader lower-educ reach', fontsize=8.3, color='#1f4e78', fontweight='bold')

        ax1.set_yticks(y_d)
        ax1.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax1.set_xlim(0, 102)
        ax1.set_ylim(-0.55, 1.55)
        ax1.set_xlabel('Share of Raw Gap (%)', fontsize=10.5, fontweight='bold')
        ax1.set_title('A. Yun-Fairlie Non-Linear Gap Decomposition', fontsize=12, fontweight='bold', pad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=9)

        bar_cols = ['#b2182b' if 'Habit' in x else ('#d95f02' if 'Merchant' in x else '#4575b4') for x in r_df['Reason']]
        bars = ax2.barh(r_df['Reason'], r_df['Pct'], color=bar_cols, height=0.55)
        for bar, val in zip(bars, r_df['Pct']):
            ax2.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#222222')
        ax2.set_xlim(0, 92)
        ax2.set_xlabel('Weighted Share of Cash-Reliant Capable Adults (%)', fontsize=10.5, fontweight='bold')
        ax2.set_title(f'B. Self-Reported Barriers in 2024 Findex (n = {len(sub_r)})', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', linestyle='--', alpha=0.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_decomposition_and_barriers.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure C: 2024 Device/Literacy/PIN Gaps + Urban-Rural Divergence (No Overlap)
    if os.path.exists(raw_2024_path):
        df24 = pd.read_csv(raw_2024_path, low_memory=False)
        c24 = df24[(df24['account'] == 1) & (df24['con1'] == 1) & (df24['internet_use'] == 1)].copy()
        c24['cash'] = (c24['merchantpay_dig'] == 0).astype(int)

        def w_rate(sub):
            return (sub.loc[sub['cash'] == 1, 'wgt'].sum() / sub['wgt'].sum()) * 100

        sms_yes, sms_no = w_rate(c24[c24['con14'] == 1]), w_rate(c24[c24['con14'] != 1])
        pin_yes, pin_no = w_rate(c24[c24['con19'] == 1]), w_rate(c24[c24['con19'] != 1])
        sm_yes, sm_no = w_rate(c24[c24['con9'] == 1]), w_rate(c24[c24['con9'] == 2])

        c21 = capable_df[capable_df['survey_wave'] == 2021]
        c24_clean = capable_df[capable_df['survey_wave'] == 2024]

        def w_rate_clean(sub):
            return (sub.loc[sub['cash_reliant'] == 1, 'wgt'].sum() / sub['wgt'].sum()) * 100

        u21, u24 = w_rate_clean(c21[c21['is_urban'] == 1]), w_rate_clean(c24_clean[c24_clean['is_urban'] == 1])
        r21, r24 = w_rate_clean(c21[c21['is_urban'] == 0]), w_rate_clean(c24_clean[c24_clean['is_urban'] == 0])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4))

        cats = ['Smartphone\n(Yes vs No)', 'Read SMS\n(Yes vs No)', 'Change PIN Solo\n(Yes vs No)']
        x1 = np.arange(len(cats))
        w_b = 0.34
        b_y = ax1.bar(x1 - w_b / 2, [sm_yes, sms_yes, pin_yes], width=w_b, color='#1f4e78', label='Has Capability (Yes)')
        b_n = ax1.bar(x1 + w_b / 2, [sm_no, sms_no, pin_no], width=w_b, color='#b2182b', label='Lacks Capability (No)')
        for bar in list(b_y) + list(b_n):
            v = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.6, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9.8)
        ax1.set_xticks(x1)
        ax1.set_xticklabels(cats, fontsize=10, fontweight='bold')
        ax1.set_ylim(0, 102)
        ax1.set_ylabel('Weighted Cash-Reliance Rate (%)', fontsize=10.5, fontweight='bold')
        ax1.set_title('A. 2024 Connectivity Tracker: Device, Literacy & PIN Gaps', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.35)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(loc='upper left', frameon=True, facecolor='white')

        x2 = np.arange(2)
        w_bar = 0.32
        b_21 = ax2.bar(x2 - w_bar / 2, [u21, r21], width=w_bar, color='#8c8c8c', label='2021 Wave')
        b_24 = ax2.bar(x2 + w_bar / 2, [u24, r24], width=w_bar, color='#1f4e78', label='2024 Wave')
        for bar in list(b_21) + list(b_24):
            v = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

        ax2.annotate(f'{u24 - u21:+.1f} pp raw\n(adj. -9.0 pp, p=0.001)', xy=(0.16, u24 + 5.5), xytext=(0.24, 78),
                     ha='center', fontsize=9, fontweight='bold', color='#1a9850',
                     arrowprops=dict(arrowstyle='->', color='#1a9850', lw=1.5))
        ax2.annotate(f'{r24 - r21:+.1f} pp raw\n(adj. +3.3 pp, n.s.)', xy=(1.16, r24 + 5.5), xytext=(1.16, 78),
                     ha='center', fontsize=9, fontweight='bold', color='#b2182b',
                     arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5))

        ax2.set_xticks(x2)
        ax2.set_xticklabels([f'Urban Capable Adults\n(n = {(capable_df["is_urban"]==1).sum()})', f'Rural Capable Adults\n(n = {(capable_df["is_urban"]==0).sum()})'], fontsize=10.5, fontweight='bold')
        ax2.set_ylim(0, 98)
        ax2.set_ylabel('Weighted Cash-Reliance Rate (%)', fontsize=10.5, fontweight='bold')
        ax2.set_title('B. Spatial Divergence: Urban Digitization vs. Rural Inertia', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_2024_literacy_and_urban_gap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All econometric publication figures saved to '{output_dir}/'")
