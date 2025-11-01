# ============================================================================
# EVALUATOR MODULE
# File: src/evaluator.py
# ============================================================================
# Evaluates model performance with comprehensive metrics

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def evaluate_models(rf_cal, gb_cal, lr_cal, X_test, y_test, scaler):
    """
    Evaluate all models on test set
    
    Parameters:
    -----------
    rf_cal, gb_cal, lr_cal : calibrated models
        Calibrated classifier objects
    X_test : array-like
        Test features (scaled)
    y_test : array-like
        Test targets
    scaler : RobustScaler
        Fitted scaler
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    
    print("\n  Random Forest:")
    acc_rf, p_rf, r_rf, f1_rf, auc_rf, pred_rf, proba_rf = evaluate_model(
        rf_cal, X_test, y_test, "Random Forest"
    )
    
    print("\n  Gradient Boosting:")
    acc_gb, p_gb, r_gb, f1_gb, auc_gb, pred_gb, proba_gb = evaluate_model(
        gb_cal, X_test, y_test, "Gradient Boosting"
    )
    
    print("\n  Logistic Regression:")
    acc_lr, p_lr, r_lr, f1_lr, auc_lr, pred_lr, proba_lr = evaluate_model(
        lr_cal, X_test, y_test, "Logistic Regression"
    )
    
    print("\n  Ensemble (Average Probability):")
    proba_ens = (proba_rf + proba_gb + proba_lr) / 3
    pred_ens = (proba_ens > 0.5).astype(int)
    
    acc_ens = accuracy_score(y_test, pred_ens)
    p_ens = precision_score(y_test, pred_ens, zero_division=0)
    r_ens = recall_score(y_test, pred_ens, zero_division=0)
    f1_ens = f1_score(y_test, pred_ens, zero_division=0)
    auc_ens = roc_auc_score(y_test, proba_ens)
    
    print(f"    Accuracy:  {acc_ens:.4f}")
    print(f"    Precision: {p_ens:.4f}")
    print(f"    Recall:    {r_ens:.4f}")
    print(f"    F1 Score:  {f1_ens:.4f}")
    print(f"    AUC-ROC:   {auc_ens:.4f}")
    
    results = {
        'rf_accuracy': acc_rf,
        'rf_precision': p_rf,
        'rf_recall': r_rf,
        'rf_f1': f1_rf,
        'rf_auc': auc_rf,
        'gb_accuracy': acc_gb,
        'gb_precision': p_gb,
        'gb_recall': r_gb,
        'gb_f1': f1_gb,
        'gb_auc': auc_gb,
        'lr_accuracy': acc_lr,
        'lr_precision': p_lr,
        'lr_recall': r_lr,
        'lr_f1': f1_lr,
        'lr_auc': auc_lr,
        'ensemble_accuracy': acc_ens,
        'ensemble_precision': p_ens,
        'ensemble_recall': r_ens,
        'ensemble_f1': f1_ens,
        'ensemble_auc': auc_ens,
        'top_features': []
    }
    
    return results


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate single model and print metrics"""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")
    
    return accuracy, precision, recall, f1, auc, y_pred, y_proba
