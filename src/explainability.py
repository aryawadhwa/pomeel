"""
Explainability & Publication Visualization Module:
1. Human-readable TreeSHAP visualizations (Global Importance by Category,
   Directional Beeswarm, Non-Linear Age Inflection Curve, Wave Shift 2021 vs 2024,
   and Persona Waterfall Case Studies).
2. Econometric Publication Figures (Wave-Separated AME Forest Plot, Fairlie Decomposition
   & Stated Barriers Panel, and 2024 Digital Literacy & Urban-Rural Divergence Panel).
"""

import os
from typing import Dict, Any, Optional

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
    'emergency_possible': 'Emergency Funds Buffer',
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
    'can_read_sms': 'Can Read SMS / WhatsApp',
    'can_send_sms': 'Can Send SMS / WhatsApp',
    'has_phone_lock': 'Phone Has PIN / Biometric Lock',
    'family_phone_rules': 'Family Sets Rules on Phone',
    'received_scam_msg': 'Received Scam SMS / Call',
    'frequent_net_use': 'Daily Internet Usage',
}

FEATURE_CATEGORIES: Dict[str, str] = {
    'has_debit_card': 'Financial Access & Depth',
    'borrowed_any': 'Financial Access & Depth',
    'saved_any': 'Financial Access & Depth',
    'saved_formal': 'Financial Access & Depth',
    'emergency_possible': 'Financial Access & Depth',
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
    'age_over_40': 'Socio-Demographic & Habit',
    'emp_in': 'Socio-Demographic & Habit',
    'is_urban': 'Socio-Demographic & Habit',
    'wave_x_urban': 'Temporal Wave Shift (2021→2024)',
    'wave_2024': 'Temporal Wave Shift (2021→2024)',
    'wave_x_age40': 'Temporal Wave Shift (2021→2024)',
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
    Executes complete TreeSHAP interpretability pipeline with human-readable labels
    and generates clean, publication-grade visualizations.
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

    # 1. Global Feature Importance (Top 15, Category-Colored + Value Labels)
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

    # 2. Directional Summary (Beeswarm) with Unclipped Title & Directional Callout
    fig = plt.figure(figsize=(11, 7.5))
    shap.plots.beeswarm(shap_pretty, max_display=14, show=False)
    ax = plt.gca()
    ax.set_title('How Features Push Predictions: Digital Pay vs. Cash Reliance (TreeSHAP)', fontsize=13, fontweight='bold', pad=16)
    ax.set_xlabel('SHAP Value   |   ← Pushes Toward Digital Pay       Pushes Toward Cash Reliance →', fontsize=10.5, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Dependence Plot: Age Non-Linearity with Binned Trend & Age-40 Inflection Line
    age_idx = feature_names.index('age')
    cash_idx_col = feature_names.index('cash_inflow_intensity') if 'cash_inflow_intensity' in feature_names else 0
    ages = X_test['age'].values
    age_shap = shap_values.values[:, age_idx]
    cash_vals = X_test.iloc[:, cash_idx_col].values

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sc = ax.scatter(ages, age_shap, c=cash_vals, cmap='coolwarm', alpha=0.75, s=38, edgecolors='white', linewidth=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Cash-Inflow Intensity (0–3 Channels)', fontsize=10, fontweight='bold')

    # Smoothed binned mean curve
    bins = np.arange(15, 72, 5)
    bin_centers, bin_means = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (ages >= b0) & (ages < b1)
        if m.sum() >= 3:
            bin_centers.append((b0 + b1) / 2)
            bin_means.append(age_shap[m].mean())
    ax.plot(bin_centers, bin_means, color='#111111', linewidth=2.5, marker='o', markersize=5, label='Binned Mean Trend')

    ax.axhline(0, color='#666666', linestyle=':', linewidth=1.2)
    ax.axvline(40, color='#b2182b', linestyle='--', linewidth=1.8, label='Inflection Threshold (Age 40)')
    ax.axvspan(40, ages.max() + 2, color='#b2182b', alpha=0.06)
    ax.text(41.2, age_shap.max() * 0.85, 'Steep Post-40 Habit\nInertia Slope →', color='#b2182b', fontsize=10, fontweight='bold')

    ax.set_xlabel('Respondent Age (Years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('SHAP Impact on Cash Reliance (Log-Odds)', fontsize=11, fontweight='bold')
    ax.set_title('Age Inflection Curve: Non-Linear Jump in Cash Reliance Past Age 40', fontsize=13, fontweight='bold', pad=12)
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

    # 4. Local Waterfall Case Studies (Human-Readable Labels)
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
    }


def generate_econometric_figures(
    capable_df: pd.DataFrame,
    raw_2024_path: str = 'data/raw/Findex_Microdata_2025_updateIndia.csv',
    output_dir: str = 'figures',
) -> None:
    """
    Generates three publication-grade economics figures from the computed CSV outputs:
    1. econ_ames_forest_plot.png (2021 vs 2024 Average Marginal Effects in pp)
    2. econ_decomposition_and_barriers.png (Fairlie Decomposition + fin25e4 Stated Barriers)
    3. econ_2024_literacy_and_urban_gap.png (SMS Literacy + Urban vs Rural Wave Shift)
    """
    os.makedirs(output_dir, exist_ok=True)
    _clean_style()

    # Figure A: Econometric AME Forest Plot (2021 vs 2024)
    ames_path = os.path.join(output_dir, 'econometric_ames_by_wave.csv')
    if os.path.exists(ames_path):
        ames_df = pd.read_csv(ames_path)
        key_feats = ['has_debit_card', 'wage_bank', 'saved_any', 'borrowed_any', 'educ', 'inc_q', 'is_urban', 'female', 'wage_cash']
        sub = ames_df[ames_df['Feature'].isin(key_feats)].copy()
        sub['order'] = sub['AME_2024']
        sub = sub.sort_values('order', ascending=True).reset_index(drop=True)

        y = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        ax.axvline(0, color='#333333', linestyle='--', linewidth=1.2)

        ax.scatter(sub['AME_2021'] * 100, y - 0.14, color='#7f7f7f', s=75, marker='s', label='2021 Wave (AME in pp)', zorder=3)
        ax.scatter(sub['AME_2024'] * 100, y + 0.14, color='#1f4e78', s=85, marker='o', label='2024 Wave (AME in pp)', zorder=4)

        for idx, row in sub.iterrows():
            v21 = row['AME_2021'] * 100
            v24 = row['AME_2024'] * 100
            ax.plot([v21, v24], [idx - 0.14, idx + 0.14], color='#bbbbbb', linewidth=1.2, zorder=2)
            ax.text(v24 + (0.8 if v24 >= 0 else -0.8), idx + 0.18, f'{v24:+.1f} pp',
                    ha='left' if v24 >= 0 else 'right', va='center', fontsize=8.5, color='#1f4e78', fontweight='bold')

        ax.set_yticks(y)
        ax.set_yticklabels([_label_for(f) for f in sub['Feature']], fontsize=10)
        ax.set_xlim(-46, 20)
        ax.set_ylim(-0.8, len(sub) + 0.45)
        ax.set_xlabel('Average Marginal Effect on Cash Reliance (Percentage Points)', fontsize=11, fontweight='bold')
        ax.set_title('How Drivers Evolved (2021 → 2024): Survey-Weighted Marginal Effects', fontsize=13, fontweight='bold', pad=18)
        ax.axvspan(0, 20, color='#d73027', alpha=0.04)
        ax.axvspan(-46, 0, color='#1a9850', alpha=0.04)
        ax.text(-23, len(sub) - 0.05, '← Reduces Cash Reliance', color='#1a9850', fontsize=9.5, fontweight='bold', ha='center', va='center')
        ax.text(10, len(sub) - 0.05, 'Increases Cash Reliance →', color='#b2182b', fontsize=9.5, fontweight='bold', ha='center', va='center')
        ax.grid(axis='x', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_ames_forest_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure B: Fairlie Decomposition + 2024 Stated Reasons (fin25e4)
    decomp_path = os.path.join(output_dir, 'decomposition_results.csv')
    if os.path.exists(decomp_path) and os.path.exists(raw_2024_path):
        dec = pd.read_csv(decomp_path)
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
        r_vals = []
        for code, lbl in reason_map.items():
            pct = (sub_r.loc[sub_r['fin25e4'] == code, 'wgt'].sum() / wt) * 100
            r_vals.append((lbl, pct))
        r_df = pd.DataFrame(r_vals, columns=['Reason', 'Pct']).sort_values('Pct', ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={'width_ratios': [1.0, 1.15]})

        # Panel A: Fairlie Decomposition
        labels = ['2021 → 2024 Drop\n(−5.8 pp gap)', 'Gender Usage Gap\n(+14.9 pp female gap)']
        explained = dec['Pct_Explained_by_Endowments'].values
        unexplained = dec['Pct_Unexplained_Behavioral'].values
        y_d = np.arange(len(labels))
        ax1.barh(y_d, explained, height=0.45, color='#1f4e78', label='Explained by Endowments\n(Debit Card, Bank Wages, Educ)')
        ax1.barh(y_d, unexplained, left=explained, height=0.45, color='#d95f02', label='Unexplained / Behavioral Shift\n(Habits, Soundboxes, Household Norms)')
        for i in range(len(labels)):
            ax1.text(explained[i] / 2, i, f'{explained[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
            ax1.text(explained[i] + unexplained[i] / 2, i, f'{unexplained[i]:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
        ax1.set_yticks(y_d)
        ax1.set_yticklabels(labels, fontsize=10, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(-0.55, 1.55)
        ax1.set_xlabel('Share of Total Gap (%)', fontsize=10.5, fontweight='bold', labelpad=8)
        ax1.set_title('A. Fairlie Non-Linear Gap Decomposition', fontsize=12, fontweight='bold', pad=10)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=8.8)

        # Panel B: 2024 Stated Reasons (fin25e4)
        bar_cols = ['#b2182b' if 'Habit' in x else ('#d95f02' if 'Merchant' in x else '#4575b4') for x in r_df['Reason']]
        bars = ax2.barh(r_df['Reason'], r_df['Pct'], color=bar_cols, height=0.55)
        for bar, val in zip(bars, r_df['Pct']):
            ax2.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold', color='#222222')
        ax2.set_xlim(0, 92)
        ax2.set_xlabel('Weighted Share of Cash-Reliant Capable Adults (%)', fontsize=10.5, fontweight='bold')
        ax2.set_title('B. Self-Reported Barriers in 2024 Findex (n = 711)', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='x', linestyle='--', alpha=0.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_decomposition_and_barriers.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure C: 2024 Functional Literacy (con14/con18) & Urban-Rural Wave Divergence
    if os.path.exists(raw_2024_path):
        df24 = pd.read_csv(raw_2024_path, low_memory=False)
        c24 = df24[(df24['account'] == 1) & (df24['con1'] == 1) & (df24['internet_use'] == 1)].copy()
        c24['cash'] = (c24['merchantpay_dig'] == 0).astype(int)

        def w_rate(sub):
            return (sub.loc[sub['cash'] == 1, 'wgt'].sum() / sub['wgt'].sum()) * 100

        sms_yes = w_rate(c24[c24['con14'] == 1])
        sms_no = w_rate(c24[c24['con14'] == 2])
        lock_yes = w_rate(c24[c24['con18'] == 1])
        lock_no = w_rate(c24[c24['con18'] == 2])

        # Urban vs Rural 2021 vs 2024 weighted cash rates
        c21 = capable_df[capable_df['survey_wave'] == 2021]
        c24_clean = capable_df[capable_df['survey_wave'] == 2024]
        def w_rate_clean(sub):
            return (sub.loc[sub['cash_reliant'] == 1, 'wgt'].sum() / sub['wgt'].sum()) * 100

        u21 = w_rate_clean(c21[c21['is_urban'] == 1])
        u24 = w_rate_clean(c24_clean[c24_clean['is_urban'] == 1])
        r21 = w_rate_clean(c21[c21['is_urban'] == 0])
        r24 = w_rate_clean(c24_clean[c24_clean['is_urban'] == 0])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

        # Panel A: 2024 Functional Literacy & Security Hygiene
        cats = ['Can Read SMS\n(Yes)', 'Cannot Read SMS\n(No)', 'Has Phone Lock\n(Yes)', 'No Phone Lock\n(No)']
        vals = [sms_yes, sms_no, lock_yes, lock_no]
        cols = ['#1f4e78', '#b2182b', '#1f4e78', '#d95f02']
        bars1 = ax1.bar(cats, vals, color=cols, width=0.55)
        for bar, v in zip(bars1, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10.5)
        ax1.set_ylim(0, 96)
        ax1.set_ylabel('Weighted Cash-Reliance Rate (%)', fontsize=10.5, fontweight='bold')
        ax1.set_title('A. 2024 Connectivity Tracker: Effort Expectancy Gap', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.35)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Panel B: Urban vs Rural Divergence (2021 -> 2024)
        x = np.arange(2)
        w_bar = 0.32
        b_21 = ax2.bar(x - w_bar / 2, [u21, r21], width=w_bar, color='#8c8c8c', label='2021 Wave')
        b_24 = ax2.bar(x + w_bar / 2, [u24, r24], width=w_bar, color='#1f4e78', label='2024 Wave')
        for bar in list(b_21) + list(b_24):
            v = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

        ax2.annotate(f'{u24 - u21:+.1f} pp raw\n(adj. -9.0 pp, p=0.001)', xy=(0.16, u24 + 5.5), xytext=(0.24, 79),
                     ha='center', fontsize=9, fontweight='bold', color='#1a9850',
                     arrowprops=dict(arrowstyle='->', color='#1a9850', lw=1.5))
        ax2.annotate(f'{r24 - r21:+.1f} pp raw\n(adj. +3.3 pp, n.s.)', xy=(1.16, r24 + 5.5), xytext=(1.16, 79),
                     ha='center', fontsize=9, fontweight='bold', color='#b2182b',
                     arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5))

        ax2.set_xticks(x)
        ax2.set_xticklabels(['Urban Capable Adults\n(n = 971)', 'Rural Capable Adults\n(n = 1,201)'], fontsize=10.5, fontweight='bold')
        ax2.set_ylim(0, 92)
        ax2.set_ylabel('Weighted Cash-Reliance Rate (%)', fontsize=10.5, fontweight='bold')
        ax2.set_title('B. Spatial Divergence: Urban Digitization vs. Rural Inertia', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.35)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(loc='upper left', frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'econ_2024_literacy_and_urban_gap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"All econometric publication figures saved to '{output_dir}/'")
