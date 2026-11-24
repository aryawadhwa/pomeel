"""
Econometric & Behavioral Audit Suite:
1. Wave-Separated Survey-Weighted Average Marginal Effects (2021 vs. 2024 vs. Pooled)
2. Non-Linear Blinder-Oaxaca / Fairlie Decomposition (Temporal 2021->2024 & Gender Gap)
3. Subgroup Heterogeneity AMEs (Urban/Rural, Male/Female, Age <40 vs >=40)
4. 2024 Digital Connectivity Tracker Deep-Dive (con11/14/18/20/21) & Merchant Refusal Logit (fin25e4==1)
"""

import os
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
]


def _fit_weighted_logit_ame(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'cash_reliant') -> pd.DataFrame:
    """Fits survey-weighted Logit (GLM Binomial) and returns Average Marginal Effects (dy/dx)."""
    # Drop zero-variance columns inside subgroup if any
    active_cols = [c for c in feature_cols if df[c].nunique() > 1]
    X_glm = sm.add_constant(df[active_cols].astype(float), has_constant='add')
    y = df[target_col].astype(float).values
    w = df['wgt'].astype(float).values

    # Fit GLM for predicted probabilities and Logit model for get_margeff
    logit_mod = sm.Logit(y, X_glm)
    # Use GLM Binomial with freq_weights to capture survey weights cleanly
    glm_res = sm.GLM(y, X_glm, family=sm.families.Binomial(), freq_weights=w).fit()
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
    # Handle statsmodels column name variation for upper CI
    for col in list(margeff.columns):
        if 'Hi' in col or 'High' in col:
            margeff = margeff.rename(columns={col: 'CI_High'})
    margeff.index.name = 'Feature'
    return margeff.reset_index()


def run_wave_separated_ames(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Computes Average Marginal Effects separately for 2021, 2024, and Pooled."""
    os.makedirs(output_dir, exist_ok=True)
    df21 = capable_df[capable_df['survey_wave'] == 2021].copy()
    df24 = capable_df[capable_df['survey_wave'] == 2024].copy()

    ame21 = _fit_weighted_logit_ame(df21, CORE_GLM_FEATURES)[['Feature', 'AME_dy_dx', 'p_value']].rename(
        columns={'AME_dy_dx': 'AME_2021', 'p_value': 'p_2021'}
    )
    ame24 = _fit_weighted_logit_ame(df24, CORE_GLM_FEATURES)[['Feature', 'AME_dy_dx', 'p_value']].rename(
        columns={'AME_dy_dx': 'AME_2024', 'p_value': 'p_2024'}
    )
    ame_pool = _fit_weighted_logit_ame(capable_df, CORE_GLM_FEATURES + ['wave_2024'])[
        ['Feature', 'AME_dy_dx', 'p_value']
    ].rename(columns={'AME_dy_dx': 'AME_Pooled', 'p_value': 'p_Pooled'})

    merged = ame_pool.merge(ame21, on='Feature', how='left').merge(ame24, on='Feature', how='left')
    merged.to_csv(os.path.join(output_dir, 'econometric_ames_by_wave.csv'), index=False)
    return merged


def _nonlinear_oaxaca_fairlie(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    feature_cols: List[str],
    group_label: str,
) -> Dict[str, Any]:
    """
    Non-linear Blinder-Oaxaca (Yun / Fairlie) decomposition for Logit models:
    Decomposes weighted mean outcome difference (Mean_A - Mean_B) into:
    - Endowment (Explained) component: E[F(X_A * beta_ref)] - E[F(X_B * beta_ref)]
    - Coefficient / Behavioral (Unexplained) component: remainder
    Uses pooled survey-weighted Logit coefficients (beta_ref) as the reference structure.
    """
    df_pool = pd.concat([df_a, df_b], ignore_index=True)
    active_cols = [c for c in feature_cols if df_pool[c].nunique() > 1]

    X_pool = sm.add_constant(df_pool[active_cols].astype(float), has_constant='add')
    y_pool = df_pool['cash_reliant'].astype(float).values
    w_pool = df_pool['wgt'].astype(float).values
    res_pool = sm.GLM(y_pool, X_pool, family=sm.families.Binomial(), freq_weights=w_pool).fit()

    X_a = sm.add_constant(df_a[active_cols].astype(float), has_constant='add')
    X_b = sm.add_constant(df_b[active_cols].astype(float), has_constant='add')
    w_a = df_a['wgt'].astype(float).values
    w_b = df_b['wgt'].astype(float).values

    mean_a = np.average(df_a['cash_reliant'].values, weights=w_a)
    mean_b = np.average(df_b['cash_reliant'].values, weights=w_b)
    raw_gap = mean_a - mean_b

    # Counterfactual probabilities under pooled structural coefficients
    p_a_ref = res_pool.predict(X_a)
    p_b_ref = res_pool.predict(X_b)
    endowment_effect = np.average(p_a_ref, weights=w_a) - np.average(p_b_ref, weights=w_b)
    behavioral_effect = raw_gap - endowment_effect

    pct_explained = (endowment_effect / raw_gap * 100) if abs(raw_gap) > 1e-8 else 0.0
    pct_unexplained = 100.0 - pct_explained

    return {
        'Comparison': group_label,
        'Rate_Group_A': mean_a,
        'Rate_Group_B': mean_b,
        'Raw_Gap_A_minus_B': raw_gap,
        'Endowment_Explained_Gap': endowment_effect,
        'Behavioral_Unexplained_Gap': behavioral_effect,
        'Pct_Explained_by_Endowments': pct_explained,
        'Pct_Unexplained_Behavioral': pct_unexplained,
    }


def run_fairlie_oaxaca_decomposition(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Runs Temporal (2021 vs 2024) and Gender (Female vs Male) non-linear decompositions."""
    os.makedirs(output_dir, exist_ok=True)
    df21 = capable_df[capable_df['survey_wave'] == 2021].copy()
    df24 = capable_df[capable_df['survey_wave'] == 2024].copy()
    decomp_time = _nonlinear_oaxaca_fairlie(
        df21, df24, CORE_GLM_FEATURES, 'Temporal Shift: 2021 (A) vs 2024 (B)'
    )

    df_f = capable_df[capable_df['female'] == 1].copy()
    df_m = capable_df[capable_df['female'] == 0].copy()
    gender_feats = [c for c in CORE_GLM_FEATURES if c != 'female'] + ['wave_2024']
    decomp_gender = _nonlinear_oaxaca_fairlie(
        df_f, df_m, gender_feats, 'Gender Gap: Female (A) vs Male (B)'
    )

    res_df = pd.DataFrame([decomp_time, decomp_gender])
    res_df.to_csv(os.path.join(output_dir, 'decomposition_results.csv'), index=False)
    return res_df


def run_subgroup_heterogeneity(capable_df: pd.DataFrame, output_dir: str = 'figures') -> pd.DataFrame:
    """Computes AMEs across Urban/Rural, Female/Male, and Age <40 vs >=40."""
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


def run_2024_tracker_and_merchant_refusal_audit(
    raw_2024_path: str = 'data/raw/Findex_Microdata_2025_updateIndia.csv',
    output_dir: str = 'figures',
) -> Dict[str, Any]:
    """
    1. Runs 2024 Digital Connectivity Tracker Deep-Dive (N=1,335) incorporating
       can_read_sms (con14), can_send_sms (con11), has_phone_lock (con18),
       family_phone_rules (con20), received_scam_msg (con21), frequent_net_use (con26).
    2. Runs Stated Merchant-Refusal Logit (fin25e4 == 1) on cash-reliant capable adults.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(raw_2024_path):
        print(f"[INFO] Skipping 2024 Connectivity Tracker deep-dive ({raw_2024_path} not found).")
        return {}

    df24 = pd.read_csv(raw_2024_path, low_memory=False)
    c24 = df24[(df24['account'] == 1) & (df24['con1'] == 1) & (df24['internet_use'] == 1)].copy()

    # Build clean 2024 deep-dive dataframe
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

    # New 2024 Digital Connectivity Tracker variables (strictly leak-free!)
    d['can_read_sms'] = (c24['con14'] == 1).astype(int)
    d['can_send_sms'] = (c24['con11'] == 1).astype(int)
    d['has_phone_lock'] = (c24['con18'] == 1).astype(int)
    d['family_phone_rules'] = (c24['con20'] == 1).astype(int)
    d['received_scam_msg'] = (c24['con21'] == 1).astype(int)
    d['frequent_net_use'] = (c24['con26'] == 1).astype(int)

    tracker_cols = [c for c in d.columns if c not in ('cash_reliant', 'wgt')]

    # 1a. Survey-weighted AMEs on 2024 Tracker features
    ame_24_tracker = _fit_weighted_logit_ame(d, tracker_cols)
    ame_24_tracker.to_csv(os.path.join(output_dir, 'tracker_2024_ames.csv'), index=False)

    # 1b. XGBoost 5x3 CV + TreeSHAP on 2024 Tracker
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

    # 2. Stated Merchant-Refusal Logit (fin25e4 == 1 among cash-reliant capable adults)
    sub_cash = c24[c24['fin25e4'].isin([1, 2, 3, 4, 5])].copy()
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
        'tracker_2024_shap': shap_24_df,
        'merchant_refusal_ames': merchant_ame,
        'merchant_refusal_n': len(m_df),
    }


def run_full_econometric_suite(capable_df: pd.DataFrame, output_dir: str = 'figures') -> Dict[str, Any]:
    """Runs all four econometric pillars and prints concise summaries."""
    print("\n[Econometrics 1/4] Computing Survey-Weighted Average Marginal Effects (2021 vs 2024 vs Pooled)...")
    ames_wave = run_wave_separated_ames(capable_df, output_dir=output_dir)

    print("[Econometrics 2/4] Running Non-Linear Blinder-Oaxaca / Fairlie Decomposition...")
    decomp = run_fairlie_oaxaca_decomposition(capable_df, output_dir=output_dir)
    print(decomp[['Comparison', 'Rate_Group_A', 'Rate_Group_B', 'Raw_Gap_A_minus_B', 'Pct_Explained_by_Endowments', 'Pct_Unexplained_Behavioral']].to_string(index=False))

    print("\n[Econometrics 3/4] Running Subgroup Heterogeneity AMEs (Urban/Rural, Gender, Age)...")
    hetero = run_subgroup_heterogeneity(capable_df, output_dir=output_dir)

    print("[Econometrics 4/4] Running 2024 Connectivity Tracker & Merchant-Refusal (fin25e4==1) Audit...")
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
        'tracker_2024': tracker_res,
    }
