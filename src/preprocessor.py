"""
Preprocessor module for isolating the 'Capable-but-Cash' cohort,
strictly leak-free feature engineering, survey weight handling,
and stratified train/test partitioning.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


BASELINE_FEATURE_COLS: List[str] = [
    'age',
    'female',
    'educ',
    'inc_q',
    'is_urban',
    'wage_cash',
    'gov_cash',
    'emergency_possible',
    'wave_2024',
]

RAW_MAIN_EFFECTS_COLS: List[str] = [
    'age',
    'female',
    'educ',
    'inc_q',
    'is_urban',
    'emp_in',
    'wave_2024',
    'wage_cash',
    'wage_bank',
    'gov_cash',
    'gov_bank',
    'ag_cash',
    'has_debit_card',
    'saved_formal',
    'saved_any',
    'borrowed_any',
    'emergency_possible',
    'emergency_not_difficult',
]

ENHANCED_FEATURE_COLS: List[str] = [
    # Socio-demographics
    'age',
    'female',
    'educ',
    'inc_q',
    'is_urban',
    'emp_in',
    'wave_2024',
    # Cash-Out Loop & Income Inflows (upstream, strictly leak-free)
    'wage_cash',
    'wage_bank',
    'gov_cash',
    'gov_bank',
    'ag_cash',
    # Upstream Structural Financial Instruments & Resilience
    'has_debit_card',
    'saved_formal',
    'saved_any',
    'borrowed_any',
    'emergency_possible',
    'emergency_not_difficult',
    # Engineered Domain & Temporal Interactions
    'cash_inflow_intensity',
    'bank_inflow_intensity',
    'wave_x_urban',
    'wave_x_female',
    'educ_x_inc',
]

FEATURE_BLOCKS: Dict[str, List[str]] = {
    'Block 1: Socio-Demographics Only': [
        'age', 'female', 'educ', 'inc_q', 'is_urban', 'emp_in', 'wave_2024',
    ],
    'Block 2: + Income Rails (Cash-Out Loop)': [
        'age', 'female', 'educ', 'inc_q', 'is_urban', 'emp_in', 'wave_2024',
        'wage_cash', 'wage_bank', 'gov_cash', 'gov_bank', 'ag_cash',
        'cash_inflow_intensity', 'bank_inflow_intensity',
    ],
    'Block 3: + Financial Depth & Resilience (No Card)': [
        'age', 'female', 'educ', 'inc_q', 'is_urban', 'emp_in', 'wave_2024',
        'wage_cash', 'wage_bank', 'gov_cash', 'gov_bank', 'ag_cash',
        'cash_inflow_intensity', 'bank_inflow_intensity',
        'saved_formal', 'saved_any', 'borrowed_any',
        'emergency_possible', 'emergency_not_difficult',
    ],
    'Block 4: Full Model (+ Debit Card & Interactions)': list(ENHANCED_FEATURE_COLS),
}


def _coalesce_flag(df: pd.DataFrame, checks: list) -> pd.Series:
    """Evaluates a list of (column, condition_fn) pairs across waves cleanly."""
    mask = pd.Series(False, index=df.index)
    for col, fn in checks:
        if col in df.columns:
            mask = mask | fn(df[col]).fillna(False)
    return mask.astype(int)


def harmonize_findex_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonizes raw World Bank Findex question codes (2021 & 2024) to strictly
    leak-free upstream predictors and domain interactions.

    EXCLUDED (to prevent simultaneous digital-payment target leakage):
    pay_utilities, anydigpayment, fin4, fin5, fin8, fin14a.
    """
    clean_df = pd.DataFrame(index=df.index)

    # Track survey wave (2021 vs 2024)
    if 'survey_wave' in df.columns:
        clean_df['survey_wave'] = df['survey_wave']
    elif 'year' in df.columns:
        clean_df['survey_wave'] = df['year']
    else:
        clean_df['survey_wave'] = 2021

    if 'wave_2024' in df.columns:
        clean_df['wave_2024'] = pd.to_numeric(df['wave_2024'], errors='coerce').fillna(0).astype(int)
    else:
        clean_df['wave_2024'] = (clean_df['survey_wave'] == 2024).astype(int)

    # Survey sampling weights
    clean_df['wgt'] = pd.to_numeric(df['wgt'], errors='coerce').fillna(1.0) if 'wgt' in df.columns else 1.0

    # 1. Core Capability Indicators (Account + Mobile + Internet)
    clean_df['account'] = _coalesce_flag(df, [('account', lambda s: s == 1), ('fin2', lambda s: s == 1)])
    clean_df['mobileowner'] = _coalesce_flag(df, [('mobileowner', lambda s: s == 1), ('con1', lambda s: s == 1)])
    clean_df['internet'] = _coalesce_flag(
        df,
        [
            ('internet', lambda s: s == 1),
            ('internetaccess', lambda s: s == 1),
            ('internet_use', lambda s: s == 1),
        ],
    )

    # 2. Target Variable (Cash-Reliant at In-Store Merchants)
    clean_df['merchantpay_dig'] = _coalesce_flag(df, [('merchantpay_dig', lambda s: s == 1)])
    if 'cash_reliant' in df.columns:
        clean_df['cash_reliant'] = pd.to_numeric(df['cash_reliant'], errors='coerce').fillna(0).astype(int)
    else:
        clean_df['cash_reliant'] = (clean_df['merchantpay_dig'] == 0).astype(int)

    # 3. Socio-Demographics (1=Female, 2=Urban in Findex)
    clean_df['female'] = _coalesce_flag(df, [('female', lambda s: s == 1)])
    clean_df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(35.0).clip(15, 90) if 'age' in df.columns else 35.0
    clean_df['educ'] = pd.to_numeric(df['educ'], errors='coerce').fillna(2).clip(1, 3).astype(int) if 'educ' in df.columns else 2
    clean_df['inc_q'] = pd.to_numeric(df['inc_q'], errors='coerce').fillna(3).clip(1, 5).astype(int) if 'inc_q' in df.columns else 3

    if 'is_urban' in df.columns:
        clean_df['is_urban'] = pd.to_numeric(df['is_urban'], errors='coerce').fillna(0).astype(int)
    else:
        clean_df['is_urban'] = _coalesce_flag(
            df,
            [('urbanicity_f2f', lambda s: s == 2), ('urbanicity', lambda s: s == 2)],
        )

    clean_df['emp_in'] = _coalesce_flag(df, [('emp_in', lambda s: s == 1)])

    # 4. Verified Cash-Out Loop & Income Inflow Channels (World Bank Standardized)
    clean_df['wage_cash'] = _coalesce_flag(df, [('wage_cash', lambda s: s == 1), ('receive_wages', lambda s: s == 2)])
    clean_df['wage_bank'] = _coalesce_flag(df, [('wage_bank', lambda s: s == 1), ('receive_wages', lambda s: s == 1)])
    clean_df['gov_cash'] = _coalesce_flag(df, [('gov_cash', lambda s: s == 1), ('receive_transfers', lambda s: s == 2)])
    clean_df['gov_bank'] = _coalesce_flag(df, [('gov_bank', lambda s: s == 1), ('receive_transfers', lambda s: s == 1)])
    clean_df['ag_cash'] = _coalesce_flag(df, [('ag_cash', lambda s: s == 1), ('receive_agriculture', lambda s: s == 2)])

    # 5. Upstream Structural Financial Instruments & Resilience
    clean_df['has_debit_card'] = _coalesce_flag(df, [('has_debit_card', lambda s: s == 1), ('fin2', lambda s: s == 1)])
    clean_df['saved_formal'] = _coalesce_flag(df, [('saved_formal', lambda s: s == 1), ('fin17a', lambda s: s == 1)])
    clean_df['saved_any'] = _coalesce_flag(df, [('saved_any', lambda s: s == 1), ('saved', lambda s: s == 1)])
    clean_df['borrowed_any'] = _coalesce_flag(df, [('borrowed_any', lambda s: s == 1), ('borrowed', lambda s: s == 1)])
    clean_df['emergency_possible'] = _coalesce_flag(
        df,
        [('emergency_possible', lambda s: s == 1), ('fin24', lambda s: (s >= 1) & (s <= 6))],
    )
    clean_df['emergency_not_difficult'] = _coalesce_flag(
        df,
        [('emergency_not_difficult', lambda s: s == 1), ('fin24a', lambda s: s.isin([2, 3]))],
    )

    # 6. Engineered Behavioral & Temporal Interactions
    clean_df['cash_inflow_intensity'] = clean_df['wage_cash'] + clean_df['gov_cash'] + clean_df['ag_cash']
    clean_df['bank_inflow_intensity'] = clean_df['wage_bank'] + clean_df['gov_bank']
    clean_df['wave_x_urban'] = clean_df['wave_2024'] * clean_df['is_urban']
    clean_df['wave_x_female'] = clean_df['wave_2024'] * clean_df['female']
    clean_df['age_over_40'] = (clean_df['age'] >= 40).astype(int)
    clean_df['wave_x_age40'] = clean_df['wave_2024'] * clean_df['age_over_40']
    clean_df['educ_x_inc'] = clean_df['educ'] * clean_df['inc_q']

    return clean_df


def filter_capable_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolates the 'Capable' segment: respondents possessing an active bank account,
    mobile phone ownership, and internet access.
    """
    df_clean = harmonize_findex_features(df)
    condition = (df_clean['account'] == 1) & (df_clean['mobileowner'] == 1) & (df_clean['internet'] == 1)
    capable_df = df_clean[condition].copy()
    capable_df['cash_reliant'] = (capable_df['merchantpay_dig'] == 0).astype(int)
    return capable_df


def prepare_features_and_targets(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Extracts strictly leak-free feature matrices (both baseline & enhanced),
    target vector, and survey sampling weights with stratified train/test split.
    """
    df = harmonize_findex_features(df)
    feature_cols = list(ENHANCED_FEATURE_COLS)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[feature_cols].copy()
    y = df['cash_reliant'].values
    weights = df['wgt'].values if 'wgt' in df.columns else np.ones(len(df))

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return {
        'X_full': X.reset_index(drop=True),
        'y_full': y,
        'w_full': weights,
        'X_train': X.iloc[train_idx].reset_index(drop=True),
        'X_test': X.iloc[test_idx].reset_index(drop=True),
        'y_train': y[train_idx],
        'y_test': y[test_idx],
        'w_train': weights[train_idx],
        'w_test': weights[test_idx],
        'feature_names': feature_cols,
        'baseline_features': list(BASELINE_FEATURE_COLS),
        'raw_main_features': list(RAW_MAIN_EFFECTS_COLS),
        'feature_blocks': FEATURE_BLOCKS,
        'full_capable_df': df,
    }


if __name__ == '__main__':
    from data_loader import load_findex_data

    raw_df = load_findex_data()
    capable_df = filter_capable_cohort(raw_df)
    print(f"Total surveyed: {len(raw_df):,}")
    print(f"Capable segment: {len(capable_df):,} ({len(capable_df)/len(raw_df)*100:.2f}%)")
    data = prepare_features_and_targets(capable_df)
    print(f"Enhanced leak-free features ({len(data['feature_names'])}):", data['feature_names'])
