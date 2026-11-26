"""
Predictive Modeling & Validation Module:
Implements:
1. Survey-Weighted Econometric Logistic Regression (GLM with Odds Ratios & CIs)
2. Non-Linear Spline + One-Hot Logistic Regression
3. Constrained, Regularized Gradient Boosted Trees (XGBoost & LightGBM)
4. Diverse 4-Family Stacking Ensemble (SplineLogit + XGB + LGBM + ExtraTrees)
5. Platt Sigmoid Calibration & Out-of-Fold Decision Threshold Optimization
6. Repeated Stratified K-Fold CV (5x3 = 15 evaluations) preserving survey weights (wgt)
"""

import warnings
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

set_config(enable_metadata_routing=True)


def make_plain_logit(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Standard linear Logistic Regression with scaling."""
    sc = StandardScaler().set_fit_request(sample_weight=False)
    lr = LogisticRegression(C=C, max_iter=1000, random_state=random_state).set_fit_request(sample_weight=True)
    return Pipeline([('sc', sc), ('clf', lr)])


def make_spline_logit(C: float = 0.3, random_state: int = 42) -> Pipeline:
    """Logistic Regression with B-splines on age and one-hot encoding on educ/inc_q."""
    sp = SplineTransformer(n_knots=4, degree=3).set_fit_request(sample_weight=False)
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
    pre = ColumnTransformer(
        [
            ('age_sp', sp, ['age']),
            ('cat', ohe, ['educ', 'inc_q']),
        ],
        remainder='passthrough',
    )
    sc = StandardScaler().set_fit_request(sample_weight=False)
    lr = LogisticRegression(C=C, max_iter=1000, random_state=random_state).set_fit_request(sample_weight=True)
    return Pipeline([('pre', pre), ('sc', sc), ('clf', lr)])


def make_xgb_classifier(random_state: int = 42) -> XGBClassifier:
    """Constrained, regularized XGBoost calibrated for ~2,000-row survey data."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        min_child_weight=12,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=2.5,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=1,
    ).set_fit_request(sample_weight=True)


def make_lgb_classifier(random_state: int = 42) -> LGBMClassifier:
    """Constrained, regularized LightGBM classifier."""
    return LGBMClassifier(
        n_estimators=200,
        max_depth=3,
        num_leaves=7,
        min_child_samples=25,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=2.5,
        random_state=random_state,
        verbose=-1,
        n_jobs=1,
    ).set_fit_request(sample_weight=True)


def make_extratrees_classifier(random_state: int = 42) -> ExtraTreesClassifier:
    """Regularized ExtraTrees for ensemble diversity."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=10,
        max_features=0.70,
        random_state=random_state,
        n_jobs=1,
    ).set_fit_request(sample_weight=True)


def make_stacked_ensemble(random_state: int = 42) -> StackingClassifier:
    """4-family StackingClassifier trained on out-of-fold predictions."""
    estimators = [
        ('logit', make_spline_logit(random_state=random_state)),
        ('xgb', make_xgb_classifier(random_state=random_state)),
        ('lgb', make_lgb_classifier(random_state=random_state)),
        ('et', make_extratrees_classifier(random_state=random_state)),
    ]
    meta = LogisticRegression(C=1.0, max_iter=1000).set_fit_request(sample_weight=True)
    return StackingClassifier(estimators=estimators, final_estimator=meta, cv=5, n_jobs=1)


def make_calibrated_stack(random_state: int = 42) -> CalibratedClassifierCV:
    """Platt Sigmoid calibration wrapped around the 4-family stack."""
    return CalibratedClassifierCV(
        make_stacked_ensemble(random_state=random_state),
        method='sigmoid',
        cv=3,
    ).set_fit_request(sample_weight=True)


def weighted_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes survey-weighted Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_w = float(np.sum(w))
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (y_prob >= b0) & (y_prob <= b1 if i == n_bins - 1 else y_prob < b1)
        if np.any(m):
            bw = float(np.sum(w[m]))
            acc = float(np.average(y_true[m], weights=w[m]))
            conf = float(np.average(y_prob[m], weights=w[m]))
            ece += (bw / total_w) * abs(acc - conf)
    return float(ece)


def compute_weighted_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    w: np.ndarray,
    thresh_f1: float = 0.50,
    thresh_ba: float = 0.50,
) -> Dict[str, Any]:
    """Computes survey-weighted discrimination, calibration, and thresholded metrics."""
    y_pred_05 = (y_prob >= 0.50).astype(int)
    y_pred_f1 = (y_prob >= thresh_f1).astype(int)
    y_pred_ba = (y_prob >= thresh_ba).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, y_prob, sample_weight=w),
        'pr_auc': average_precision_score(y_true, y_prob, sample_weight=w),
        'brier_score': brier_score_loss(y_true, y_prob, sample_weight=w),
        'ece': weighted_ece(y_true, y_prob, w, n_bins=10),
        'f1_05': f1_score(y_true, y_pred_05, sample_weight=w),
        'f1_opt': f1_score(y_true, y_pred_f1, sample_weight=w),
        'balanced_acc_05': balanced_accuracy_score(y_true, y_pred_05, sample_weight=w),
        'balanced_acc_opt': balanced_accuracy_score(y_true, y_pred_ba, sample_weight=w),
        'thresh_f1': thresh_f1,
        'thresh_ba': thresh_ba,
        'confusion_matrix': confusion_matrix(y_true, y_pred_f1),
    }


def find_optimal_thresholds_oof(
    model_fn: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    cv_folds: int = 5,
) -> Tuple[float, float]:
    """
    Sweeps cutoffs on out-of-fold validation predictions strictly inside the
    training set to find thresholds maximizing survey-weighted F1 and Balanced Accuracy.
    """
    oof_probs = cross_val_predict(
        model_fn(),
        X_train,
        y_train,
        cv=cv_folds,
        method='predict_proba',
        params={'sample_weight': w_train},
    )[:, 1]

    best_th_f1, best_f1 = 0.50, -1.0
    best_th_ba, best_ba = 0.50, -1.0
    for th in np.linspace(0.25, 0.75, 51):
        preds = (oof_probs >= th).astype(int)
        s_f1 = f1_score(y_train, preds, sample_weight=w_train)
        s_ba = balanced_accuracy_score(y_train, preds, sample_weight=w_train)
        if s_f1 > best_f1:
            best_f1, best_th_f1 = s_f1, float(th)
        if s_ba > best_ba:
            best_ba, best_th_ba = s_ba, float(th)
    return best_th_f1, best_th_ba


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    w_train: np.ndarray = None,
    w_test: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Fits a non-collinear survey-weighted GLM Binomial (Logit) baseline
    for Odds Ratios & 95% Confidence Intervals alongside Spline-Logit metrics.
    """
    # Drop exact algebraic sums (intensities & interaction duplicates) for well-conditioned GLM
    drop_for_glm = {'cash_inflow_intensity', 'bank_inflow_intensity', 'age_over_40', 'educ_x_inc'}
    glm_cols = [c for c in X_train.columns if c not in drop_for_glm]

    X_tr_glm = sm.add_constant(X_train[glm_cols])

    glm_model = sm.GLM(
        y_train,
        X_tr_glm,
        family=sm.families.Binomial(),
        freq_weights=w_train if w_train is not None else None,
    )
    glm_res = glm_model.fit(cov_type='HC1')

    params = glm_res.params
    conf = glm_res.conf_int()
    odds_ratios = pd.DataFrame(
        {
            'Coefficient': params,
            'Std_Error': glm_res.bse,
            'p_value': glm_res.pvalues,
            'Odds_Ratio': np.exp(np.clip(params, -20, 20)),
            'CI_lower': np.exp(np.clip(conf[0], -20, 20)),
            'CI_upper': np.exp(np.clip(conf[1], -20, 20)),
        }
    )

    # Fit Spline + One-Hot Logistic Regression for predictive benchmark
    spline_logit = make_spline_logit()
    spline_logit.fit(X_train, y_train, sample_weight=w_train)
    th_f1, th_ba = find_optimal_thresholds_oof(make_spline_logit, X_train, y_train, w_train, cv_folds=5)
    y_pred_prob = spline_logit.predict_proba(X_test)[:, 1]
    metrics = compute_weighted_metrics(y_test, y_pred_prob, w_test, thresh_f1=th_f1, thresh_ba=th_ba)

    return {
        'model': spline_logit,
        'glm_model': glm_res,
        'odds_ratios': odds_ratios,
        'metrics': metrics,
        'y_pred_prob': y_pred_prob,
        'y_pred': (y_pred_prob >= th_f1).astype(int),
    }


def train_model_with_postprocessing(
    model_fn: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    tune_thresholds: bool = True,
) -> Dict[str, Any]:
    """Fits any weighted model, tunes OOF decision thresholds, and evaluates on held-out test."""
    clf = model_fn()
    clf.fit(X_train, y_train, sample_weight=w_train)

    if tune_thresholds:
        th_f1, th_ba = find_optimal_thresholds_oof(model_fn, X_train, y_train, w_train, cv_folds=5)
    else:
        th_f1, th_ba = 0.50, 0.50

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    metrics = compute_weighted_metrics(y_test, y_pred_prob, w_test, thresh_f1=th_f1, thresh_ba=th_ba)
    return {
        'model': clf,
        'metrics': metrics,
        'y_pred_prob': y_pred_prob,
        'y_pred': (y_pred_prob >= th_f1).astype(int),
    }


def run_repeated_cv_benchmark(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    w_full: np.ndarray,
    baseline_features: List[str],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Runs 5x3 Repeated Stratified K-Fold CV (15 evaluations) on core models
    and 5-fold Stratified CV on the calibrated stack, returning both summary metrics
    and fold-level ROC-AUC arrays for paired significance testing.
    """
    warnings.filterwarnings('ignore')
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    no_card_cols = [c for c in X_full.columns if c != 'has_debit_card']

    configs = [
        ('Baseline XGBoost (9 Core Vars)', lambda: make_xgb_classifier(random_state), X_full[baseline_features], rskf, False),
        ('Ablation XGBoost (No Debit Card)', lambda: make_xgb_classifier(random_state), X_full[no_card_cols], rskf, False),
        ('Plain Linear Logit (Enhanced)', lambda: make_plain_logit(random_state=random_state), X_full, rskf, True),
        ('Spline + OneHot Logit (Enhanced)', lambda: make_spline_logit(random_state=random_state), X_full, rskf, True),
        ('Regularized XGBoost (Enhanced)', lambda: make_xgb_classifier(random_state), X_full, rskf, True),
        ('Regularized LightGBM (Enhanced)', lambda: make_lgb_classifier(random_state), X_full, rskf, True),
        ('Stacked Ensemble + Platt Cal', lambda: make_calibrated_stack(random_state), X_full, skf5, False),
    ]

    rows = []
    fold_aucs: Dict[str, np.ndarray] = {}
    for name, fn, X_sub, splitter, tune_th in configs:
        aucs, praucs, briers, eces, f1s, baccs = [], [], [], [], [], []
        for tr_idx, te_idx in splitter.split(X_sub, y_full):
            X_tr, X_te = X_sub.iloc[tr_idx], X_sub.iloc[te_idx]
            y_tr, y_te = y_full[tr_idx], y_full[te_idx]
            w_tr, w_te = w_full[tr_idx], w_full[te_idx]

            clf = fn()
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            prob_te = clf.predict_proba(X_te)[:, 1]

            if tune_th:
                th_f1, th_ba = find_optimal_thresholds_oof(fn, X_tr, y_tr, w_tr, cv_folds=3)
            else:
                th_f1, th_ba = 0.50, 0.50

            aucs.append(roc_auc_score(y_te, prob_te, sample_weight=w_te))
            praucs.append(average_precision_score(y_te, prob_te, sample_weight=w_te))
            briers.append(brier_score_loss(y_te, prob_te, sample_weight=w_te))
            eces.append(weighted_ece(y_te, prob_te, w_te, n_bins=10))
            f1s.append(f1_score(y_te, (prob_te >= th_f1).astype(int), sample_weight=w_te))
            baccs.append(balanced_accuracy_score(y_te, (prob_te >= th_ba).astype(int), sample_weight=w_te))

        fold_aucs[name] = np.array(aucs)
        rows.append(
            {
                'Model_Configuration': name,
                'ROC_AUC_Mean': np.mean(aucs),
                'ROC_AUC_Std': np.std(aucs),
                'PR_AUC_Mean': np.mean(praucs),
                'Brier_Score_Mean': np.mean(briers),
                'Weighted_ECE_Mean': np.mean(eces),
                'F1_Score_Mean': np.mean(f1s),
                'Balanced_Acc_Mean': np.mean(baccs),
            }
        )
        print(
            f"  [CV] {name:34s} | AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.3f} "
            f"| PR-AUC: {np.mean(praucs):.4f} | Brier: {np.mean(briers):.4f} "
            f"| ECE: {np.mean(eces):.4f} | F1: {np.mean(f1s):.4f}"
        )

    return pd.DataFrame(rows), fold_aucs


def _midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks for Sun & Xu (2014) fast DeLong variance estimation."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Two-sided DeLong test comparing two correlated ROC-AUCs on the same test split."""
    pos = y_true == 1
    neg = y_true == 0
    m = int(pos.sum())
    n = int(neg.sum())
    preds = np.vstack([pred_a, pred_b])
    tx = np.array([_midrank(p[pos]) for p in preds])
    ty = np.array([_midrank(p[neg]) for p in preds])
    tz = np.array([_midrank(np.concatenate([p[pos], p[neg]])) for p in preds])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    diff = float(aucs[0] - aucs[1])
    var = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var <= 1e-12:
        return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]), 'z_stat': 0.0, 'p_value': 1.0}
    z = diff / np.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]), 'z_stat': float(z), 'p_value': p}


def run_statistical_comparisons(
    fold_aucs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    test_probs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Computes paired Wilcoxon CV tests and held-out DeLong ROC-AUC tests."""
    rows = []
    cv_pairs = [
        ('Regularized XGBoost (Enhanced)', 'Baseline XGBoost (9 Core Vars)', 'Enhanced vs. 9-Var Baseline'),
        ('Regularized XGBoost (Enhanced)', 'Ablation XGBoost (No Debit Card)', 'Full Model vs. No-Debit-Card Ablation'),
        ('Regularized XGBoost (Enhanced)', 'Plain Linear Logit (Enhanced)', 'XGBoost vs. Plain Linear Logit'),
        ('Regularized XGBoost (Enhanced)', 'Spline + OneHot Logit (Enhanced)', 'XGBoost vs. Spline-Logit'),
    ]
    for m_a, m_b, label in cv_pairs:
        a, b = fold_aucs[m_a], fold_aucs[m_b]
        stat, p = stats.wilcoxon(a, b)
        rows.append({
            'Test_Type': '5x3 CV Paired Wilcoxon',
            'Comparison': label,
            'Metric_A': float(np.mean(a)),
            'Metric_B': float(np.mean(b)),
            'Diff_A_minus_B': float(np.mean(a - b)),
            'Statistic': float(stat),
            'p_value': float(p),
        })

    delong_pairs = [
        ('Stacked + Platt Cal', 'Spline-Logit (Baseline+)', 'Held-Out DeLong: Calibrated Stack vs. Spline-Logit'),
        ('XGBoost (Reg+FE)', 'Spline-Logit (Baseline+)', 'Held-Out DeLong: XGBoost vs. Spline-Logit'),
    ]
    for m_a, m_b, label in delong_pairs:
        dl = delong_roc_test(y_test, test_probs[m_a], test_probs[m_b])
        rows.append({
            'Test_Type': 'Held-Out Unweighted DeLong',
            'Comparison': label,
            'Metric_A': dl['auc_a'],
            'Metric_B': dl['auc_b'],
            'Diff_A_minus_B': dl['auc_a'] - dl['auc_b'],
            'Statistic': dl['z_stat'],
            'p_value': dl['p_value'],
        })
    return pd.DataFrame(rows)


def run_hierarchical_ablation(
    X_full: pd.DataFrame,
    y_full: np.ndarray,
    w_full: np.ndarray,
    feature_blocks: Dict[str, List[str]],
    random_state: int = 42,
) -> pd.DataFrame:
    """Runs 5x3 CV across nested theoretical feature blocks for Logit and XGBoost."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    rows = []
    for block_name, cols in feature_blocks.items():
        X_sub = X_full[cols].reset_index(drop=True)
        for model_label, fn in [
            ('Plain Logit', lambda: make_plain_logit(random_state=random_state)),
            ('XGBoost', lambda: make_xgb_classifier(random_state=random_state)),
        ]:
            aucs, praucs, briers = [], [], []
            for tr_idx, te_idx in rskf.split(X_sub, y_full):
                clf = fn()
                clf.fit(X_sub.iloc[tr_idx], y_full[tr_idx], sample_weight=w_full[tr_idx])
                p = clf.predict_proba(X_sub.iloc[te_idx])[:, 1]
                aucs.append(roc_auc_score(y_full[te_idx], p, sample_weight=w_full[te_idx]))
                praucs.append(average_precision_score(y_full[te_idx], p, sample_weight=w_full[te_idx]))
                briers.append(brier_score_loss(y_full[te_idx], p, sample_weight=w_full[te_idx]))
            rows.append({
                'Feature_Block': block_name,
                'Num_Features': len(cols),
                'Model': model_label,
                'ROC_AUC_Mean': float(np.mean(aucs)),
                'ROC_AUC_Std': float(np.std(aucs)),
                'PR_AUC_Mean': float(np.mean(praucs)),
                'Brier_Mean': float(np.mean(briers)),
            })
    return pd.DataFrame(rows)


def run_cross_wave_transfer(
    capable_df: pd.DataFrame,
    feature_cols: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluates out-of-distribution temporal generalization (2021->2024 and 2024->2021)."""
    wave_invariant = [c for c in feature_cols if 'wave' not in c]
    df21 = capable_df[capable_df['survey_wave'] == 2021].reset_index(drop=True)
    df24 = capable_df[capable_df['survey_wave'] == 2024].reset_index(drop=True)

    models = [
        ('Plain Linear Logit', lambda: make_plain_logit(random_state=random_state)),
        ('Spline + OneHot Logit', lambda: make_spline_logit(random_state=random_state)),
        ('Regularized XGBoost', lambda: make_xgb_classifier(random_state=random_state)),
        ('Regularized LightGBM', lambda: make_lgb_classifier(random_state=random_state)),
        ('Stacked + Platt Cal', lambda: make_calibrated_stack(random_state=random_state)),
    ]

    rows = []
    for direction, train_df, test_df in [
        ('Train 2021 -> Test 2024', df21, df24),
        ('Train 2024 -> Test 2021', df24, df21),
    ]:
        X_tr, y_tr, w_tr = train_df[wave_invariant], train_df['cash_reliant'].values, train_df['wgt'].values
        X_te, y_te, w_te = test_df[wave_invariant], test_df['cash_reliant'].values, test_df['wgt'].values
        for m_name, fn in models:
            clf = fn()
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            rows.append({
                'Transfer_Direction': direction,
                'Model': m_name,
                'ROC_AUC': float(roc_auc_score(y_te, prob, sample_weight=w_te)),
                'PR_AUC': float(average_precision_score(y_te, prob, sample_weight=w_te)),
                'Brier_Score': float(brier_score_loss(y_te, prob, sample_weight=w_te)),
                'Weighted_ECE': weighted_ece(y_te, prob, w_te, n_bins=10),
            })
    return pd.DataFrame(rows)


def compute_calibration_curve_data(
    y_test: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    w_test: np.ndarray,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Computes survey-weighted reliability curve bins across models."""
    rows = []
    for m_name, probs in probs_dict.items():
        quantiles = np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1))
        quantiles[0], quantiles[-1] = 0.0, 1.0
        for i in range(n_bins):
            m = (probs >= quantiles[i]) & (probs <= quantiles[i + 1] if i == n_bins - 1 else probs < quantiles[i + 1])
            if np.any(m):
                rows.append({
                    'Model': m_name,
                    'Bin': i + 1,
                    'Mean_Predicted_Prob': float(np.average(probs[m], weights=w_test[m])),
                    'Observed_Cash_Rate': float(np.average(y_test[m], weights=w_test[m])),
                    'Bin_Weight': float(np.sum(w_test[m])),
                    'Count': int(np.sum(m)),
                })
    return pd.DataFrame(rows)


def run_fairness_audit(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    w_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Audits subgroup discrimination, calibration, and error parity on held-out test split."""
    strata = [
        ('Overall Test', np.ones(len(X_test), dtype=bool)),
        ('Female', (X_test['female'] == 1).values),
        ('Male', (X_test['female'] == 0).values),
        ('Urban', (X_test['is_urban'] == 1).values),
        ('Rural', (X_test['is_urban'] == 0).values),
        ('Age < 40', (X_test['age'] < 40).values),
        ('Age >= 40', (X_test['age'] >= 40).values),
    ]
    rows = []
    y_pred = (y_prob >= threshold).astype(int)
    for label, mask in strata:
        yt, yp, pr, wt = y_test[mask], y_prob[mask], y_pred[mask], w_test[mask]
        pos = yt == 1
        neg = yt == 0
        tpr = float(np.average(pr[pos] == 1, weights=wt[pos])) if np.any(pos) else np.nan
        fpr = float(np.average(pr[neg] == 1, weights=wt[neg])) if np.any(neg) else np.nan
        rows.append({
            'Subgroup': label,
            'N': int(mask.sum()),
            'Observed_Cash_Rate': float(np.average(yt, weights=wt)),
            'Predicted_Cash_Rate': float(np.average(yp, weights=wt)),
            'ROC_AUC': float(roc_auc_score(yt, yp, sample_weight=wt)),
            'Brier_Score': float(brier_score_loss(yt, yp, sample_weight=wt)),
            'Weighted_ECE': weighted_ece(yt, yp, wt, n_bins=8),
            'TPR_Recall': tpr,
            'FPR': fpr,
        })
    return pd.DataFrame(rows)


def run_policy_counterfactual_simulation(
    calibrated_model: Any,
    capable_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Computes associational model-implied counterfactual cash-reliance rates
    on the 2024 capable cohort (n=1,335) under structural policy interventions.
    """
    d24 = capable_df[capable_df['survey_wave'] == 2024].copy().reset_index(drop=True)
    w24 = d24['wgt'].values
    base_X = d24[feature_cols].copy()
    base_rate = float(np.average(calibrated_model.predict_proba(base_X)[:, 1], weights=w24))
    obs_rate = float(np.average(d24['cash_reliant'].values, weights=w24))

    scenarios = [('Observed 2024 Weighted Rate', obs_rate, 0.0)]
    scenarios.append(('Baseline Model Prediction (2024)', base_rate, 0.0))

    # Scenario 1: Universal Onboarding Parity (Aadhaar-OTP UPI / Debit Card parity)
    X_card = base_X.copy()
    X_card['has_debit_card'] = 1
    r_card = float(np.average(calibrated_model.predict_proba(X_card)[:, 1], weights=w24))
    scenarios.append(('Scenario 1: Aadhaar-OTP Onboarding Parity (Card=1)', r_card, (r_card - base_rate) * 100))

    # Scenario 2: Cash-Out Loop Interception (Convert cash wages -> bank wages)
    X_wage = base_X.copy()
    mask_cw = X_wage['wage_cash'] == 1
    X_wage.loc[mask_cw, 'wage_cash'] = 0
    X_wage.loc[mask_cw, 'wage_bank'] = 1
    X_wage['cash_inflow_intensity'] = X_wage['wage_cash'] + X_wage['gov_cash'] + X_wage['ag_cash']
    X_wage['bank_inflow_intensity'] = X_wage['wage_bank'] + X_wage['gov_bank']
    r_wage = float(np.average(calibrated_model.predict_proba(X_wage)[:, 1], weights=w24))
    scenarios.append(('Scenario 2: Payroll Digitization (Wage Cash -> Bank)', r_wage, (r_wage - base_rate) * 100))

    # Scenario 3: Combined Onboarding + Wage Digitization
    X_both = X_wage.copy()
    X_both['has_debit_card'] = 1
    r_both = float(np.average(calibrated_model.predict_proba(X_both)[:, 1], weights=w24))
    scenarios.append(('Scenario 3: Combined Onboarding + Wage Digitization', r_both, (r_both - base_rate) * 100))

    return pd.DataFrame(scenarios, columns=['Policy_Scenario', 'Predicted_Cash_Rate', 'Change_vs_Baseline_pp'])


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Constructs a comparative held-out evaluation table across all trained models."""
    metrics = ['roc_auc', 'pr_auc', 'f1_05', 'f1_opt', 'balanced_acc_05', 'balanced_acc_opt', 'brier_score', 'ece']
    metric_labels = [
        'ROC-AUC Score',
        'PR-AUC (Precision-Recall)',
        'F1-Score (Threshold = 0.50)',
        'F1-Score (OOF-Tuned Threshold)',
        'Balanced Accuracy (Threshold = 0.50)',
        'Balanced Accuracy (OOF-Tuned)',
        'Brier Calibration Score (lower is better)',
        'Weighted ECE (lower is better)',
    ]
    data = {'Metric': metric_labels}
    for model_name, res in results_dict.items():
        data[model_name] = [res['metrics'][m] for m in metrics]
    return pd.DataFrame(data)
