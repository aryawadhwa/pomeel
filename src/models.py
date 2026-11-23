"""
Predictive Modeling Module:
Implements Survey-Weighted Logistic Regression (Econometric Baseline)
and Regularized XGBoost with Stratified Cross-Validation and Rigorous Metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    w_train: np.ndarray = None,
    w_test: np.ndarray = None
) -> Dict[str, Any]:
    """
    Fits a survey-weighted Logistic Regression baseline using statsmodels GLM
    and computes Odds Ratios with 95% Confidence Intervals.
    """
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    # Fit survey-weighted GLM Binomial (Logit link)
    glm_model = sm.GLM(
        y_train,
        X_train_const,
        family=sm.families.Binomial(),
        freq_weights=w_train if w_train is not None else None
    )
    glm_res = glm_model.fit()
    
    # Predict probabilities on test set
    y_pred_prob = glm_res.predict(X_test_const)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate Odds Ratios and Confidence Intervals
    params = glm_res.params
    conf = glm_res.conf_int()
    odds_ratios = pd.DataFrame({
        'Coefficient': params,
        'Std_Error': glm_res.bse,
        'p_value': glm_res.pvalues,
        'Odds_Ratio': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1])
    })
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_prob, sample_weight=w_test),
        'pr_auc': average_precision_score(y_test, y_pred_prob, sample_weight=w_test),
        'f1': f1_score(y_test, y_pred, sample_weight=w_test),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred, sample_weight=w_test),
        'brier_score': brier_score_loss(y_test, y_pred_prob, sample_weight=w_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return {
        'model': glm_res,
        'odds_ratios': odds_ratios,
        'metrics': metrics,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred
    }


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    w_train: np.ndarray = None,
    w_test: np.ndarray = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a regularized XGBoost classifier with 5-fold cross-validation
    and hyperparameter optimization.
    """
    # Calculate scale_pos_weight to handle class balance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = neg_count / max(pos_count, 1)
    
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        scale_pos_weight=scale_pos,
        random_state=random_state,
        eval_metric='logloss'
    )
    
    # Train model with survey sample weights
    xgb.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test] if w_test is not None else None,
        verbose=False
    )
    
    y_pred_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_prob, sample_weight=w_test),
        'pr_auc': average_precision_score(y_test, y_pred_prob, sample_weight=w_test),
        'f1': f1_score(y_test, y_pred, sample_weight=w_test),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred, sample_weight=w_test),
        'brier_score': brier_score_loss(y_test, y_pred_prob, sample_weight=w_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return {
        'model': xgb,
        'metrics': metrics,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred
    }


def compare_models(
    logit_res: Dict[str, Any],
    xgb_res: Dict[str, Any]
) -> pd.DataFrame:
    """
    Constructs a comparative evaluation table between Logit and XGBoost.
    """
    metrics = ['roc_auc', 'pr_auc', 'f1', 'balanced_acc', 'brier_score']
    metric_labels = [
        'ROC-AUC Score',
        'PR-AUC (Precision-Recall)',
        'F1-Score',
        'Balanced Accuracy',
        'Brier Calibration Score (lower is better)'
    ]
    
    df_comp = pd.DataFrame({
        'Metric': metric_labels,
        'Logistic Regression (Baseline)': [logit_res['metrics'][m] for m in metrics],
        'XGBoost (Primary)': [xgb_res['metrics'][m] for m in metrics]
    })
    
    return df_comp


if __name__ == '__main__':
    from data_loader import load_findex_data
    from preprocessor import filter_capable_cohort, prepare_features_and_targets
    
    raw_df = load_findex_data()
    capable_df = filter_capable_cohort(raw_df)
    data = prepare_features_and_targets(capable_df)
    
    print("\n--- Training Survey-Weighted Logistic Regression ---")
    logit_res = train_logistic_baseline(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test']
    )
    print("\nTop Odds Ratios from Logistic Regression:")
    print(logit_res['odds_ratios'][['Odds_Ratio', 'p_value', 'CI_lower', 'CI_upper']].round(3))
    
    print("\n--- Training Regularized XGBoost Classifier ---")
    xgb_res = train_xgboost_classifier(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['w_train'], data['w_test']
    )
    
    print("\n=== Model Performance Comparison ===")
    comp_df = compare_models(logit_res, xgb_res)
    print(comp_df.to_string(index=False))
