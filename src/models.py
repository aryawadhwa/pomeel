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
    X_te_glm = sm.add_constant(X_test[glm_cols])

    glm_model = sm.GLM(
        y_train,
        X_tr_glm,
        family=sm.families.Binomial(),
        freq_weights=w_train if w_train is not None else None,
    )
    glm_res = glm_model.fit()

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
) -> pd.DataFrame:
    """
    Runs 5x3 Repeated Stratified K-Fold CV (15 evaluations) on core single models
    and 5-fold Stratified CV on the heavy calibrated stack, reporting mean +/- std.
    """
    warnings.filterwarnings('ignore')
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    no_card_cols = [c for c in X_full.columns if c != 'has_debit_card']

    configs = [
        ('Baseline XGBoost (9 Core Vars)', lambda: make_xgb_classifier(random_state), X_full[baseline_features], rskf, False),
        ('Ablation XGBoost (No Debit Card)', lambda: make_xgb_classifier(random_state), X_full[no_card_cols], rskf, False),
        ('Spline + OneHot Logit (Enhanced)', lambda: make_spline_logit(random_state=random_state), X_full, rskf, True),
        ('Regularized XGBoost (Enhanced)', lambda: make_xgb_classifier(random_state), X_full, rskf, True),
        ('Regularized LightGBM (Enhanced)', lambda: make_lgb_classifier(random_state), X_full, rskf, True),
        ('Stacked Ensemble + Platt Cal', lambda: make_calibrated_stack(random_state), X_full, skf5, False),
    ]

    rows = []
    for name, fn, X_sub, splitter, tune_th in configs:
        aucs, praucs, briers, f1s, baccs = [], [], [], [], []
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
            f1s.append(f1_score(y_te, (prob_te >= th_f1).astype(int), sample_weight=w_te))
            baccs.append(balanced_accuracy_score(y_te, (prob_te >= th_ba).astype(int), sample_weight=w_te))

        rows.append(
            {
                'Model_Configuration': name,
                'ROC_AUC_Mean': np.mean(aucs),
                'ROC_AUC_Std': np.std(aucs),
                'PR_AUC_Mean': np.mean(praucs),
                'Brier_Score_Mean': np.mean(briers),
                'F1_Score_Mean': np.mean(f1s),
                'Balanced_Acc_Mean': np.mean(baccs),
            }
        )
        print(
            f"  [CV] {name:34s} | AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.3f} "
            f"| PR-AUC: {np.mean(praucs):.4f} | Brier: {np.mean(briers):.4f} "
            f"| F1: {np.mean(f1s):.4f} | BalAcc: {np.mean(baccs):.4f}"
        )

    return pd.DataFrame(rows)


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Constructs a comparative held-out evaluation table across all trained models."""
    metrics = ['roc_auc', 'pr_auc', 'f1_05', 'f1_opt', 'balanced_acc_05', 'balanced_acc_opt', 'brier_score']
    metric_labels = [
        'ROC-AUC Score',
        'PR-AUC (Precision-Recall)',
        'F1-Score (Threshold = 0.50)',
        'F1-Score (OOF-Tuned Threshold)',
        'Balanced Accuracy (Threshold = 0.50)',
        'Balanced Accuracy (OOF-Tuned)',
        'Brier Calibration Score (lower is better)',
    ]
    data = {'Metric': metric_labels}
    for model_name, res in results_dict.items():
        data[model_name] = [res['metrics'][m] for m in metrics]
    return pd.DataFrame(data)
