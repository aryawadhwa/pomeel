"""
Econometric & Behavioral Audit Suite:
1. Wave-Separated Survey-Weighted Average Marginal Effects (HC1 Robust SEs)
2. Order-Invariant Non-Linear Blinder-Oaxaca / Yun-Fairlie Decomposition with Bootstrap CIs
3. Subgroup Heterogeneity, Carded Sub-Cohort Isolation & Endogeneity / Age Tests
4. 2024 Digital Connectivity Tracker Deep-Dive (Smartphone con9, English SMS con15, PIN con19),
   Intensive-Margin Payment Frequency (fin25e3), Merchant Refusal Logit (fin25e4==1),
   and Cash-Reliant Behavioral Archetypes.
"""

import os
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

CORE_GLM_FEATURES: List[str] = [
    'age',
    'female',
    'educ',
    'inc_q',
    'is_urban',
    'emp_in',
    'wage_cash',
    'wage_bank',
    'gov_bank',
    'ag_cash',
    'has_debit_card',
    'saved_any',
    'borrowed_any',
    'emergency_possible',
    'emergency_not_difficult',
]


def _fit_weighted_logit_ame(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'cash_reliant') -> pd.DataFrame:
    """Fits survey-weighted Logit (GLM Binomial, HC1 robust SEs) and returns AMEs."""
    active_cols = [
        c for c in feature_cols
        if df[c].nunique() > 1 and (df[c].nunique() > 2 or int(df[c].value_counts().min()) >= 10)
    ]
    X_glm = sm.add_constant(df[active_cols].astype(float), has_constant='add')
    y = df[target_col].astype(float).values
    w = df['wgt'].astype(float).values

    glm_res = sm.GLM(y, X_glm, family=sm.families.Binomial(), freq_weights=w).fit(cov_type='HC1')
    margeff = glm_res.get_margeff(at='overall').summary_frame()
    margeff = margeff.rename(
        columns={
            'dy/dx': 'AME_dy_dx',
            'Std. Err.': 'Std_Err',
            'Pr(>|z|)': 'p_value',
            'Conf. Int. Low': 'CI_Low',
            'Cont. Int. Hi.': 'CI_High',
        }
    )
    for col in list(margeff.columns):
        if 'Hi' in col or 'High' in col:
            margeff = margeff.rename(columns={col: 'CI_High'})
    margeff.index.name = 'Feature'
    return margeff.reset_index()


def run_wave_separated_ames(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Computes HC1-robust survey-weighted AMEs (with 95% CIs) for 2021, 2024, and Pooled."""
    os.makedirs(output_dir, exist_ok=True)
    df21 = capable_df[capable_df['survey_wave'] == 2021].copy()
    df24 = capable_df[capable_df['survey_wave'] == 2024].copy()

    ame21 = _fit_weighted_logit_ame(df21, CORE_GLM_FEATURES)[
        ['Feature', 'AME_dy_dx', 'Std_Err', 'p_value', 'CI_Low', 'CI_High']
    ].rename(columns={
        'AME_dy_dx': 'AME_2021', 'Std_Err': 'SE_2021', 'p_value': 'p_2021',
        'CI_Low': 'CI_Low_2021', 'CI_High': 'CI_High_2021',
    })
    ame24 = _fit_weighted_logit_ame(df24, CORE_GLM_FEATURES)[
        ['Feature', 'AME_dy_dx', 'Std_Err', 'p_value', 'CI_Low', 'CI_High']
    ].rename(columns={
        'AME_dy_dx': 'AME_2024', 'Std_Err': 'SE_2024', 'p_value': 'p_2024',
        'CI_Low': 'CI_Low_2024', 'CI_High': 'CI_High_2024',
    })
    ame_pool = _fit_weighted_logit_ame(capable_df, CORE_GLM_FEATURES + ['wave_2024'])[
        ['Feature', 'AME_dy_dx', 'Std_Err', 'p_value', 'CI_Low', 'CI_High']
    ].rename(columns={
        'AME_dy_dx': 'AME_Pooled', 'Std_Err': 'SE_Pooled', 'p_value': 'p_Pooled',
        'CI_Low': 'CI_Low_Pooled', 'CI_High': 'CI_High_Pooled',
    })

    merged = ame_pool.merge(ame21, on='Feature', how='left').merge(ame24, on='Feature', how='left')
    merged.to_csv(os.path.join(output_dir, 'econometric_ames_by_wave.csv'), index=False)
    return merged


def _compute_yun_fairlie_once(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    active_cols: List[str],
) -> Dict[str, Any]:
    """
    Single-pass non-linear Yun (2004) / Fairlie (2005) decomposition using
    pooled survey-weighted Logit coefficients (order-invariant covariate weights).
    """
    df_pool = pd.concat([df_a, df_b], ignore_index=True)
    X_pool = sm.add_constant(df_pool[active_cols].astype(float), has_constant='add')
    y_pool = df_pool['cash_reliant'].astype(float).values
    w_pool = df_pool['wgt'].astype(float).values
    res_pool = sm.GLM(y_pool, X_pool, family=sm.families.Binomial(), freq_weights=w_pool).fit()

    X_a = sm.add_constant(df_a[active_cols].astype(float), has_constant='add')
    X_b = sm.add_constant(df_b[active_cols].astype(float), has_constant='add')
    w_a = df_a['wgt'].astype(float).values
    w_b = df_b['wgt'].astype(float).values

    mean_a = float(np.average(df_a['cash_reliant'].values, weights=w_a))
    mean_b = float(np.average(df_b['cash_reliant'].values, weights=w_b))
    raw_gap = mean_a - mean_b

    p_a_ref = res_pool.predict(X_a)
    p_b_ref = res_pool.predict(X_b)
    endowment_effect = float(np.average(p_a_ref, weights=w_a) - np.average(p_b_ref, weights=w_b))
    behavioral_effect = raw_gap - endowment_effect

    # Yun (2004) exact order-invariant detailed covariate weights
    beta = res_pool.params[active_cols].values
    mean_xa = np.average(df_a[active_cols].astype(float).values, axis=0, weights=w_a)
    mean_xb = np.average(df_b[active_cols].astype(float).values, axis=0, weights=w_b)
    linear_terms = (mean_xa - mean_xb) * beta
    denom = float(np.sum(linear_terms))
    if abs(denom) > 1e-10:
        cov_contribs = endowment_effect * (linear_terms / denom)
    else:
        cov_contribs = np.zeros_like(linear_terms)

    pct_explained = (endowment_effect / raw_gap * 100.0) if abs(raw_gap) > 1e-8 else 0.0
    return {
        'mean_a': mean_a,
        'mean_b': mean_b,
        'raw_gap': raw_gap,
        'endowment': endowment_effect,
        'behavioral': behavioral_effect,
        'pct_explained': pct_explained,
        'pct_unexplained': 100.0 - pct_explained,
        'cov_contribs': cov_contribs,
    }


def _nonlinear_oaxaca_fairlie_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    feature_cols: List[str],
    group_label: str,
    n_boot: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs Yun-Fairlie decomposition with stratified weighted bootstrap 95% CIs."""
    df_pool = pd.concat([df_a, df_b], ignore_index=True)
    active_cols = [c for c in feature_cols if df_pool[c].nunique() > 1]
    point = _compute_yun_fairlie_once(df_a, df_b, active_cols)

    rng = np.random.default_rng(random_state)
    boot_endow, boot_behav, boot_pct, boot_cov = [], [], [], []
    n_a, n_b = len(df_a), len(df_b)
    for _ in range(n_boot):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        try:
            b_res = _compute_yun_fairlie_once(df_a.iloc[idx_a], df_b.iloc[idx_b], active_cols)
            boot_endow.append(b_res['endowment'])
            boot_behav.append(b_res['behavioral'])
            boot_pct.append(b_res['pct_explained'])
            boot_cov.append(b_res['cov_contribs'])
        except Exception:
            continue

    boot_cov_arr = np.array(boot_cov)
    agg = {
        'Comparison': group_label,
        'Rate_Group_A': point['mean_a'],
        'Rate_Group_B': point['mean_b'],
        'Raw_Gap_A_minus_B': point['raw_gap'],
        'Endowment_Explained_Gap': point['endowment'],
        'Endowment_CI_Low': float(np.percentile(boot_endow, 2.5)),
        'Endowment_CI_High': float(np.percentile(boot_endow, 97.5)),
        'Behavioral_Unexplained_Gap': point['behavioral'],
        'Behavioral_CI_Low': float(np.percentile(boot_behav, 2.5)),
        'Behavioral_CI_High': float(np.percentile(boot_behav, 97.5)),
        'Pct_Explained_by_Endowments': point['pct_explained'],
        'Pct_Explained_CI_Low': float(np.percentile(boot_pct, 2.5)),
        'Pct_Explained_CI_High': float(np.percentile(boot_pct, 97.5)),
        'Pct_Unexplained_Behavioral': point['pct_unexplained'],
    }

    detail_rows = []
    for j, feat in enumerate(active_cols):
        est = float(point['cov_contribs'][j])
        share = (est / point['raw_gap'] * 100.0) if abs(point['raw_gap']) > 1e-8 else 0.0
        ci_lo = float(np.percentile(boot_cov_arr[:, j], 2.5))
        ci_hi = float(np.percentile(boot_cov_arr[:, j], 97.5))
        detail_rows.append({
            'Comparison': group_label,
            'Feature': feat,
            'Contribution_Gap_pp': est * 100.0,
            'CI_Low_pp': ci_lo * 100.0,
            'CI_High_pp': ci_hi * 100.0,
            'Share_of_Total_Gap_Pct': share,
        })

    return {'aggregate': agg, 'detailed': pd.DataFrame(detail_rows)}


def run_fairlie_oaxaca_decomposition(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Runs Temporal (2021 vs 2024) and Gender (Female vs Male) decompositions with CIs."""
    os.makedirs(output_dir, exist_ok=True)
    df21 = capable_df[capable_df['survey_wave'] == 2021].copy()
    df24 = capable_df[capable_df['survey_wave'] == 2024].copy()
    decomp_time = _nonlinear_oaxaca_fairlie_bootstrap(
        df21, df24, CORE_GLM_FEATURES, 'Temporal Shift: 2021 (A) vs 2024 (B)'
    )

    df_f = capable_df[capable_df['female'] == 1].copy()
    df_m = capable_df[capable_df['female'] == 0].copy()
    gender_feats = [c for c in CORE_GLM_FEATURES if c != 'female'] + ['wave_2024']
    decomp_gender = _nonlinear_oaxaca_fairlie_bootstrap(
        df_f, df_m, gender_feats, 'Gender Gap: Female (A) vs Male (B)'
    )

    res_df = pd.DataFrame([decomp_time['aggregate'], decomp_gender['aggregate']])
    res_df.to_csv(os.path.join(output_dir, 'decomposition_results.csv'), index=False)

    det_df = pd.concat([decomp_time['detailed'], decomp_gender['detailed']], ignore_index=True)
    det_df.to_csv(os.path.join(output_dir, 'fairlie_detailed_decomposition.csv'), index=False)
    return res_df


def run_subgroup_heterogeneity(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Computes HC1 AMEs across Urban/Rural, Female/Male, and Age <40 vs >=40."""
    os.makedirs(output_dir, exist_ok=True)
    strata = [
        ('Urban', capable_df[capable_df['is_urban'] == 1], [c for c in CORE_GLM_FEATURES if c != 'is_urban'] + ['wave_2024']),
        ('Rural', capable_df[capable_df['is_urban'] == 0], [c for c in CORE_GLM_FEATURES if c != 'is_urban'] + ['wave_2024']),
        ('Female', capable_df[capable_df['female'] == 1], [c for c in CORE_GLM_FEATURES if c != 'female'] + ['wave_2024']),
        ('Male', capable_df[capable_df['female'] == 0], [c for c in CORE_GLM_FEATURES if c != 'female'] + ['wave_2024']),
        ('Age < 40', capable_df[capable_df['age'] < 40], CORE_GLM_FEATURES + ['wave_2024']),
        ('Age >= 40', capable_df[capable_df['age'] >= 40], CORE_GLM_FEATURES + ['wave_2024']),
    ]

    all_rows = []
    for label, sub_df, feats in strata:
        ame = _fit_weighted_logit_ame(sub_df, feats)
        ame['Subgroup'] = label
        ame['N'] = len(sub_df)
        all_rows.append(ame)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(os.path.join(output_dir, 'subgroup_heterogeneity_ames.csv'), index=False)
    return out


def run_age_and_endogeneity_tests(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """
    1. Formally tests linear age vs. piecewise hinge at age 40 vs. quadratic age^2.
    2. Runs endogeneity sensitivity check excluding financial-depth instruments
       (has_debit_card, saved_any, borrowed_any).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = capable_df.copy()
    w = df['wgt'].astype(float).values
    y = df['cash_reliant'].astype(float).values

    base_cols = CORE_GLM_FEATURES + ['wave_2024']
    X_lin = sm.add_constant(df[base_cols].astype(float))
    X_hinge = X_lin.copy()
    X_hinge['age_post40_hinge'] = np.maximum(0.0, df['age'].astype(float) - 40.0)
    X_quad = X_lin.copy()
    X_quad['age_sq'] = df['age'].astype(float) ** 2

    m_lin = sm.GLM(y, X_lin, family=sm.families.Binomial(), freq_weights=w).fit(cov_type='HC1')
    m_hinge = sm.GLM(y, X_hinge, family=sm.families.Binomial(), freq_weights=w).fit(cov_type='HC1')
    m_quad = sm.GLM(y, X_quad, family=sm.families.Binomial(), freq_weights=w).fit(cov_type='HC1')

    rows = [
        {
            'Check': 'Age Spec 1: Linear Age',
            'Term': 'age',
            'Coef_or_AME': float(m_lin.params['age']),
            'Std_Err': float(m_lin.bse['age']),
            'p_value': float(m_lin.pvalues['age']),
            'LogLik': float(m_lin.llf),
        },
        {
            'Check': 'Age Spec 2: Hinge at 40',
            'Term': 'max(0, age - 40)',
            'Coef_or_AME': float(m_hinge.params['age_post40_hinge']),
            'Std_Err': float(m_hinge.bse['age_post40_hinge']),
            'p_value': float(m_hinge.pvalues['age_post40_hinge']),
            'LogLik': float(m_hinge.llf),
        },
        {
            'Check': 'Age Spec 3: Quadratic Age^2',
            'Term': 'age^2',
            'Coef_or_AME': float(m_quad.params['age_sq']),
            'Std_Err': float(m_quad.bse['age_sq']),
            'p_value': float(m_quad.pvalues['age_sq']),
            'LogLik': float(m_quad.llf),
        },
    ]

    # Endogeneity Sensitivity: Full vs. Exogenous-Only (excluding debit card, saved, borrowed)
    exog_cols = [c for c in base_cols if c not in {'has_debit_card', 'saved_any', 'borrowed_any'}]
    ame_full = _fit_weighted_logit_ame(df, base_cols).set_index('Feature')
    ame_exog = _fit_weighted_logit_ame(df, exog_cols).set_index('Feature')
    for feat in ['female', 'educ', 'inc_q', 'wage_cash', 'wage_bank', 'ag_cash', 'age', 'is_urban']:
        rows.append({
            'Check': 'Endogeneity Sensitivity (Full AME)',
            'Term': feat,
            'Coef_or_AME': float(ame_full.loc[feat, 'AME_dy_dx']),
            'Std_Err': float(ame_full.loc[feat, 'Std_Err']),
            'p_value': float(ame_full.loc[feat, 'p_value']),
            'LogLik': float(m_lin.llf),
        })
        rows.append({
            'Check': 'Endogeneity Sensitivity (No Depth Vars AME)',
            'Term': feat,
            'Coef_or_AME': float(ame_exog.loc[feat, 'AME_dy_dx']),
            'Std_Err': float(ame_exog.loc[feat, 'Std_Err']),
            'p_value': float(ame_exog.loc[feat, 'p_value']),
            'LogLik': np.nan,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, 'robustness_age_and_endogeneity.csv'), index=False)
    return out


def run_carded_cash_audit(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """
    Isolates the 'Carded Capable' sub-cohort (has_debit_card == 1) to evaluate
    pure behavioral, wage-rail, and gender frictions free of card-onboarding bottlenecks.
    """
    os.makedirs(output_dir, exist_ok=True)
    carded = capable_df[capable_df['has_debit_card'] == 1].copy()
    feats = [c for c in CORE_GLM_FEATURES if c != 'has_debit_card'] + ['wave_2024']
    ame_carded = _fit_weighted_logit_ame(carded, feats)
    ame_carded['N_Carded'] = len(carded)
    ame_carded['Weighted_Cash_Rate_Carded'] = float(np.average(carded['cash_reliant'], weights=carded['wgt']))
    ame_carded.to_csv(os.path.join(output_dir, 'carded_subcohort_ames.csv'), index=False)

    # Archetype shares inside cash-reliant capable cohort
    cash_only = capable_df[capable_df['cash_reliant'] == 1].copy()
    wc = cash_only['wgt'].values
    arch_rows = [
        ('Total Cash-Reliant Capable Adults', len(cash_only), 100.0),
        ('Lacks Debit / ATM Card (Onboarding Bottleneck)', int((cash_only['has_debit_card'] == 0).sum()), float(np.average(cash_only['has_debit_card'] == 0, weights=wc) * 100)),
        ('Owns Debit Card Yet Still Cash-Only (Pure Behavioral/Merchant Friction)', int((cash_only['has_debit_card'] == 1).sum()), float(np.average(cash_only['has_debit_card'] == 1, weights=wc) * 100)),
        ('Rural Residents', int((cash_only['is_urban'] == 0).sum()), float(np.average(cash_only['is_urban'] == 0, weights=wc) * 100)),
        ('Female Adults (Gender Usage Gap)', int((cash_only['female'] == 1).sum()), float(np.average(cash_only['female'] == 1, weights=wc) * 100)),
        ('Adults Aged >= 40 (Older Cohort)', int((cash_only['age'] >= 40).sum()), float(np.average(cash_only['age'] >= 40, weights=wc) * 100)),
        ('Receives Wages in Cash (Cash-Out Loop)', int((cash_only['wage_cash'] == 1).sum()), float(np.average(cash_only['wage_cash'] == 1, weights=wc) * 100)),
    ]
    pd.DataFrame(arch_rows, columns=['Cash_Reliant_Segment', 'Unweighted_N', 'Weighted_Share_Pct']).to_csv(
        os.path.join(output_dir, 'cash_reliant_archetypes.csv'), index=False
    )
    return ame_carded


def run_2024_tracker_and_merchant_refusal_audit(
    raw_2024_path: str = 'data/raw/Findex_Microdata_2025_updateIndia.csv',
    output_dir: str = 'figures',
) -> Dict[str, Any]:
    """
    1. 2024 Digital Connectivity Tracker Deep-Dive (N=1,335) incorporating smartphone
       vs. basic phone (con9), SMS literacy (con14/con15/con16), PIN self-efficacy
       (con18/con19), phone autonomy (con20), scams (con21), internet recency (con24/con26),
       data packs (con27), and emergency difficulty (fin24a).
    2. Smartphone-only (con9 == 1) robustness regression.
    3. Intensive-margin digital merchant payment frequency Ordered Logit (fin25e3).
    4. Stated Merchant-Refusal Logit (fin25e4 == 1) on cash-reliant capable adults.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(raw_2024_path):
        print(f"[INFO] Skipping 2024 Connectivity Tracker deep-dive ({raw_2024_path} not found).")
        return {}

    df24 = pd.read_csv(raw_2024_path, low_memory=False)
    c24 = df24[(df24['account'] == 1) & (df24['con1'] == 1) & (df24['internet_use'] == 1)].copy()

    d = pd.DataFrame(index=c24.index)
    d['cash_reliant'] = (c24['merchantpay_dig'] == 0).astype(int)
    d['wgt'] = pd.to_numeric(c24['wgt'], errors='coerce').fillna(1.0)
    d['age'] = pd.to_numeric(c24['age'], errors='coerce').fillna(35.0).clip(15, 90)
    d['female'] = (c24['female'] == 1).astype(int)
    d['educ'] = pd.to_numeric(c24['educ'], errors='coerce').fillna(2).clip(1, 3).astype(int)
    d['inc_q'] = pd.to_numeric(c24['inc_q'], errors='coerce').fillna(3).clip(1, 5).astype(int)
    d['is_urban'] = (c24['urbanicity'] == 2).astype(int)
    d['emp_in'] = (c24['emp_in'] == 1).astype(int)
    d['wage_cash'] = (c24['receive_wages'] == 2).astype(int)
    d['wage_bank'] = (c24['receive_wages'] == 1).astype(int)
    d['gov_bank'] = (c24['receive_transfers'] == 1).astype(int)
    d['has_debit_card'] = (c24['fin2'] == 1).astype(int)
    d['saved_any'] = (c24['saved'] == 1).astype(int)
    d['borrowed_any'] = (c24['borrowed'] == 1).astype(int)
    d['emergency_not_difficult'] = c24['fin24a'].isin([2, 3]).astype(int)

    # Rich 2024 Digital Connectivity Tracker indicators (strictly leak-free!)
    d['is_smartphone'] = (c24['con9'] == 1).astype(int)
    d['can_read_sms'] = (c24['con14'] == 1).astype(int)
    d['can_read_english_sms'] = (c24['con15'] == 1).astype(int)
    d['can_send_sms'] = (c24['con16'] == 1).astype(int)
    d['has_phone_lock'] = (c24['con18'] == 1).astype(int)
    d['can_change_pin_solo'] = (c24['con19'] == 1).astype(int)
    d['family_phone_rules'] = (c24['con20'] == 1).astype(int)
    d['received_scam_msg'] = (c24['con21'] == 1).astype(int)
    d['recent_net_7d'] = (c24['con24'] == 1).astype(int)
    d['frequent_net_use'] = (c24['con26'] == 1).astype(int)
    d['buys_data_pack'] = (c24['con27'] == 1).astype(int)

    tracker_cols = [c for c in d.columns if c not in ('cash_reliant', 'wgt')]

    # 1a. Survey-weighted HC1 AMEs on 2024 Tracker features
    ame_24_tracker = _fit_weighted_logit_ame(d, tracker_cols)
    ame_24_tracker.to_csv(os.path.join(output_dir, 'tracker_2024_ames.csv'), index=False)

    # 1b. Smartphone-only (con9 == 1) robustness AMEs
    d_sm = d[d['is_smartphone'] == 1].copy()
    sm_cols = [c for c in tracker_cols if c != 'is_smartphone']
    ame_24_sm = _fit_weighted_logit_ame(d_sm, sm_cols)
    ame_24_sm['N_Smartphone'] = len(d_sm)
    ame_24_sm['Weighted_Cash_Rate_Smartphone'] = float(np.average(d_sm['cash_reliant'], weights=d_sm['wgt']))
    ame_24_sm.to_csv(os.path.join(output_dir, 'tracker_2024_smartphone_ames.csv'), index=False)

    # 1c. XGBoost 5x3 CV + TreeSHAP on 2024 Tracker
    X24 = d[tracker_cols]
    y24 = d['cash_reliant'].values
    w24 = d['wgt'].values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    aucs, praucs = [], []
    for tr, te in rskf.split(X24, y24):
        clf = XGBClassifier(
            n_estimators=200, max_depth=3, min_child_weight=10,
            learning_rate=0.03, subsample=0.75, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.5, random_state=42, eval_metric='logloss', n_jobs=1,
        )
        clf.fit(X24.iloc[tr], y24[tr], sample_weight=w24[tr])
        p = clf.predict_proba(X24.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y24[te], p, sample_weight=w24[te]))
        praucs.append(average_precision_score(y24[te], p, sample_weight=w24[te]))

    xgb_full = XGBClassifier(
        n_estimators=200, max_depth=3, min_child_weight=10,
        learning_rate=0.03, subsample=0.75, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.5, random_state=42, eval_metric='logloss', n_jobs=1,
    )
    xgb_full.fit(X24, y24, sample_weight=w24)
    sv = shap.TreeExplainer(xgb_full)(X24)
    shap_24_df = pd.DataFrame({
        'Feature': tracker_cols,
        'Mean_Absolute_SHAP': np.abs(sv.values).mean(axis=0),
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    shap_24_df.to_csv(os.path.join(output_dir, 'tracker_2024_shap.csv'), index=False)

    # 2. Intensive Margin Analysis (fin25e3: 1=Weekly, 2=Monthly, 3=<Monthly among digital payers)
    sub_int = d.loc[c24['fin25e3'].isin([1, 2, 3])].copy()
    # Recode so higher = more frequent digital payment: 0=<Monthly, 1=Monthly, 2=Weekly
    sub_int['freq_score'] = (4 - c24.loc[sub_int.index, 'fin25e3']).astype(int) - 1
    int_cols = ['age', 'female', 'educ', 'inc_q', 'is_urban', 'wage_bank', 'has_debit_card', 'can_read_sms']
    ord_mod = OrderedModel(sub_int['freq_score'], sub_int[int_cols].astype(float), distr='logit')
    ord_res = ord_mod.fit(method='bfgs', disp=False)
    int_df = pd.DataFrame({
        'Feature': int_cols,
        'Ordered_Logit_Coef': [ord_res.params[c] for c in int_cols],
        'Std_Err': [ord_res.bse[c] for c in int_cols],
        'Odds_Ratio': [np.exp(ord_res.params[c]) for c in int_cols],
        'p_value': [ord_res.pvalues[c] for c in int_cols],
        'N_Digital_Payers': len(sub_int),
    })
    int_df.to_csv(os.path.join(output_dir, 'intensive_margin_fin25e3.csv'), index=False)

    # 3. Stated Merchant-Refusal Logit (fin25e4 == 1 among cash-reliant capable adults)
    sub_cash = c24[(c24['merchantpay_dig'] == 0) & (c24['fin25e4'].isin([1, 2, 3, 4, 5]))].copy()
    sub_cash['merchant_cash_only'] = (sub_cash['fin25e4'] == 1).astype(int)
    m_df = pd.DataFrame(index=sub_cash.index)
    m_df['merchant_cash_only'] = sub_cash['merchant_cash_only']
    m_df['wgt'] = pd.to_numeric(sub_cash['wgt'], errors='coerce').fillna(1.0)
    m_df['age'] = pd.to_numeric(sub_cash['age'], errors='coerce').fillna(35.0).clip(15, 90)
    m_df['female'] = (sub_cash['female'] == 1).astype(int)
    m_df['educ'] = pd.to_numeric(sub_cash['educ'], errors='coerce').fillna(2).clip(1, 3).astype(int)
    m_df['inc_q'] = pd.to_numeric(sub_cash['inc_q'], errors='coerce').fillna(3).clip(1, 5).astype(int)
    m_df['is_urban'] = (sub_cash['urbanicity'] == 2).astype(int)
    m_df['wage_cash'] = (sub_cash['receive_wages'] == 2).astype(int)
    m_df['has_debit_card'] = (sub_cash['fin2'] == 1).astype(int)

    m_cols = ['age', 'female', 'educ', 'inc_q', 'is_urban', 'wage_cash', 'has_debit_card']
    merchant_ame = _fit_weighted_logit_ame(m_df, m_cols, target_col='merchant_cash_only')
    merchant_ame.to_csv(os.path.join(output_dir, 'merchant_refusal_logit.csv'), index=False)

    return {
        'tracker_2024_cv_auc': float(np.mean(aucs)),
        'tracker_2024_cv_auc_std': float(np.std(aucs)),
        'tracker_2024_cv_prauc': float(np.mean(praucs)),
        'tracker_2024_ames': ame_24_tracker,
        'tracker_2024_smartphone_ames': ame_24_sm,
        'tracker_2024_shap': shap_24_df,
        'intensive_margin': int_df,
        'merchant_refusal_ames': merchant_ame,
        'merchant_refusal_n': len(m_df),
    }


def run_full_econometric_suite(capable_df: pd.DataFrame, output_dir: str = 'figures') -> Dict[str, Any]:
    """Runs all econometric pillars and prints concise summaries."""
    print("\n[Econometrics 1/6] Computing HC1-Robust Survey-Weighted AMEs (2021 vs 2024 vs Pooled)...")
    ames_wave = run_wave_separated_ames(capable_df, output_dir=output_dir)

    print("[Econometrics 2/6] Running Yun-Fairlie Non-Linear Decomposition with Bootstrap CIs...")
    decomp = run_fairlie_oaxaca_decomposition(capable_df, output_dir=output_dir)
    print(decomp[['Comparison', 'Rate_Group_A', 'Rate_Group_B', 'Raw_Gap_A_minus_B', 'Pct_Explained_by_Endowments', 'Pct_Unexplained_Behavioral']].to_string(index=False))

    print("\n[Econometrics 3/6] Running Subgroup Heterogeneity AMEs (Urban/Rural, Gender, Age)...")
    hetero = run_subgroup_heterogeneity(capable_df, output_dir=output_dir)

    print("[Econometrics 4/6] Running Age Nonlinearity & Endogeneity Sensitivity Checks...")
    robustness = run_age_and_endogeneity_tests(capable_df, output_dir=output_dir)

    print("[Econometrics 5/6] Auditing Carded Sub-Cohort (has_debit_card == 1) & Cash-Reliant Archetypes...")
    carded_ame = run_carded_cash_audit(capable_df, output_dir=output_dir)

    print("[Econometrics 6/6] Running 2024 Connectivity Tracker, Smartphone Robustness, Intensive Margin & Merchant Refusal...")
    tracker_res = run_2024_tracker_and_merchant_refusal_audit(output_dir=output_dir)
    if tracker_res:
        print(
            f"  -> 2024 Tracker XGBoost 5x3 CV ROC-AUC: {tracker_res['tracker_2024_cv_auc']:.4f} "
            f"+/- {tracker_res['tracker_2024_cv_auc_std']:.4f} (PR-AUC: {tracker_res['tracker_2024_cv_prauc']:.4f})"
        )

    return {
        'ames_by_wave': ames_wave,
        'decomposition': decomp,
        'heterogeneity': hetero,
        'robustness': robustness,
        'carded_ame': carded_ame,
        'tracker_2024': tracker_res,
    }
