# ============================================================================
# LOGISTIC REGRESSION - ECONOMIC CRISIS PREDICTION
# Independent Project - No Dependencies
# ============================================================================
# File: main.py (for Linear Regression project)
"""
Economic Crisis Prediction - Logistic Regression Model
Author: Fardin
Date: November 1, 2025
Model: Logistic Regression (Standalone)
Status: Production Ready
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA GENERATION
# ============================================================================

def generate_economic_data(n_countries=8, n_years=27):
    """Generate synthetic economic data with crisis patterns"""
    
    crisis_periods = {
        'United States': ['2008-Q1', '2008-Q2', '2008-Q3', '2008-Q4', '2009-Q1', '2020-Q1', '2020-Q2'],
        'Thailand': ['1997-Q3', '1997-Q4', '1998-Q1', '1998-Q2'],
        'Indonesia': ['1997-Q3', '1997-Q4', '1998-Q1', '1998-Q2', '1998-Q3'],
        'South Korea': ['1997-Q4', '1998-Q1', '1998-Q2'],
        'Malaysia': ['1997-Q3', '1997-Q4', '1998-Q1'],
        'Greece': ['2010-Q1', '2010-Q2', '2010-Q3', '2011-Q1', '2011-Q2'],
        'Spain': ['2008-Q3', '2008-Q4', '2009-Q1', '2009-Q2'],
        'Italy': ['2008-Q3', '2008-Q4', '2009-Q1', '2020-Q1']
    }
    
    countries = list(crisis_periods.keys())
    years = list(range(1995, 1995 + n_years))
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    data_records = []
    
    for country in countries:
        for year in years:
            for quarter in quarters:
                date_str = "{0}-{1}".format(year, quarter)
                is_crisis = 1 if date_str in crisis_periods.get(country, []) else 0
                
                if is_crisis:
                    gdp_growth = np.random.normal(-2.5, 3.0)
                    unemployment = np.clip(np.random.normal(9.0, 2.5), 0, 20)
                    inflation = np.random.normal(2.5, 2.0)
                    debt_to_gdp = np.clip(np.random.normal(85, 15), 0, 200)
                    interest_rate = np.clip(np.random.normal(3.5, 2.0), 0, 15)
                    fdi_change = np.random.normal(-8, 5)
                    currency_depreciation = np.random.normal(12, 8)
                    credit_growth = np.random.normal(-5, 8)
                else:
                    gdp_growth = np.random.normal(3.5, 2.0)
                    unemployment = np.clip(np.random.normal(5.5, 1.5), 0, 20)
                    inflation = np.random.normal(2.0, 1.0)
                    debt_to_gdp = np.clip(np.random.normal(60, 20), 0, 200)
                    interest_rate = np.clip(np.random.normal(2.5, 1.5), 0, 15)
                    fdi_change = np.random.normal(5, 8)
                    currency_depreciation = np.random.normal(2, 5)
                    credit_growth = np.random.normal(8, 6)
                
                gdp_credit = gdp_growth * credit_growth
                debt_interest = (debt_to_gdp / 100.0) * interest_rate
                curr_fdi = abs(currency_depreciation) * abs(fdi_change)
                
                record = {
                    'GDP_Growth': round(gdp_growth, 2),
                    'Unemployment': round(unemployment, 2),
                    'Inflation': round(inflation, 2),
                    'Debt_to_GDP': round(debt_to_gdp, 2),
                    'Interest_Rate': round(interest_rate, 2),
                    'FDI_Change': round(fdi_change, 2),
                    'Currency_Depreciation': round(currency_depreciation, 2),
                    'Credit_Growth': round(credit_growth, 2),
                    'GDP_Credit_Int': round(gdp_credit, 2),
                    'Debt_Interest_Int': round(debt_interest, 2),
                    'Currency_FDI_Int': round(curr_fdi, 2),
                    'Crisis': is_crisis
                }
                
                data_records.append(record)
    
    df = pd.DataFrame(data_records)
    return df


# ============================================================================
# SECTION 2: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for Logistic Regression model"""
    
    print("="*100)
    print(" "*20 + "LOGISTIC REGRESSION - ECONOMIC CRISIS PREDICTION")
    print(" "*25 + "Independent Project - Version 3.0")
    print("="*100)
    
    np.random.seed(42)
    
    # Step 1: Generate Data
    print("\n[STEP 1] Generating Economic Data...")
    df = generate_economic_data()
    print("  ✓ Generated {0} samples".format(len(df)))
    print("  ✓ Crisis samples: {0}".format(df['Crisis'].sum()))
    print("  ✓ Features: {0}".format(len(df.columns) - 1))
    
    # Step 2: Prepare Data
    print("\n[STEP 2] Preparing Data...")
    X = df.drop('Crisis', axis=1).values
    y = df['Crisis'].values
    feature_names = df.drop('Crisis', axis=1).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print("  ✓ Training samples: {0}".format(len(X_train)))
    print("  ✓ Test samples: {0}".format(len(X_test)))
    print("  ✓ Train crisis rate: {0:.2%}".format(y_train.mean()))
    print("  ✓ Test crisis rate: {0:.2%}".format(y_test.mean()))
    
    # Step 3: Scale Data
    print("\n[STEP 3] Scaling Features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✓ RobustScaler applied")
    
    # Step 4: Train Model
    print("\n[STEP 4] Training Logistic Regression Model...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    print("  ✓ Model trained successfully")
    
    # Step 5: Calibrate Model
    print("\n[STEP 5] Calibrating Probabilities (Platt Scaling)...")
    lr_calibrated = CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
    lr_calibrated.fit(X_train_scaled, y_train)
    print("  ✓ Calibration complete")
    
    # Step 6: Make Predictions
    print("\n[STEP 6] Making Predictions...")
    y_pred_train = lr_calibrated.predict(X_train_scaled)
    y_pred_test = lr_calibrated.predict(X_test_scaled)
    y_proba_train = lr_calibrated.predict_proba(X_train_scaled)[:, 1]
    y_proba_test = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
    print("  ✓ Predictions generated")
    
    # Step 7: Evaluate Model
    print("\n[STEP 7] Evaluating Model Performance...")
    print("\n  TRAINING SET METRICS:")
    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec = precision_score(y_train, y_pred_train, zero_division=0)
    train_rec = recall_score(y_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    train_auc = roc_auc_score(y_train, y_proba_train)
    
    print("    Accuracy:  {0:.4f}".format(train_acc))
    print("    Precision: {0:.4f}".format(train_prec))
    print("    Recall:    {0:.4f}".format(train_rec))
    print("    F1 Score:  {0:.4f}".format(train_f1))
    print("    AUC-ROC:   {0:.4f}".format(train_auc))
    
    print("\n  TEST SET METRICS:")
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    print("    Accuracy:  {0:.4f}".format(test_acc))
    print("    Precision: {0:.4f}".format(test_prec))
    print("    Recall:    {0:.4f}".format(test_rec))
    print("    F1 Score:  {0:.4f}".format(test_f1))
    print("    AUC-ROC:   {0:.4f}".format(test_auc))
    
    # Step 8: Confusion Matrix
    print("\n[STEP 8] Confusion Matrix Analysis...")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    print("    True Negatives:  {0}".format(tn))
    print("    False Positives: {0}".format(fp))
    print("    False Negatives: {0}".format(fn))
    print("    True Positives:  {0}".format(tp))
    print("    Specificity:     {0:.4f}".format(tn / (tn + fp)))
    
    # Step 9: Cross-Validation
    print("\n[STEP 9] Cross-Validation (5-fold)...")
    cv_scores = cross_val_score(lr_calibrated, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print("    CV AUC Scores: {0}".format([round(s, 4) for s in cv_scores]))
    print("    Mean AUC:      {0:.4f} (+/- {1:.4f})".format(cv_scores.mean(), cv_scores.std()))
    
    # Step 10: Feature Coefficients
    print("\n[STEP 10] Feature Importance (Coefficients)...")
    coefficients = lr_model.coef_[0]
    feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    print("    Top 5 Features:")
    for i, (feat, coef) in enumerate(feature_importance[:5], 1):
        print("      {0}. {1}: {2:.6f}".format(i, feat, coef))
    
    # Step 11: Sample Predictions
    print("\n[STEP 11] Sample Predictions...")
    sample_data = np.array([[3.5, 5.5, 2.0, 60.0, 2.5, 5.0, 2.0, 8.0, 28.0, 1.5, 10.0]])
    sample_scaled = scaler.transform(sample_data)
    sample_pred = lr_calibrated.predict(sample_scaled)[0]
    sample_proba = lr_calibrated.predict_proba(sample_scaled)[0, 1]
    print("    Healthy Economy Prediction:")
    print("      Probability of Crisis: {0:.4f}".format(sample_proba))
    print("      Prediction: {0}".format("CRISIS" if sample_pred == 1 else "NO CRISIS"))
    
    # Step 12: Summary
    print("\n" + "="*100)
    print("LOGISTIC REGRESSION MODEL - SUMMARY")
    print("="*100)
    print("  Model Type:         Logistic Regression")
    print("  Solver:             LBFGS")
    print("  Class Weights:      Balanced")
    print("  Calibration:        Platt Scaling (Sigmoid)")
    print("  ")
    print("  Performance (Test Set):")
    print("    Accuracy:         {0:.4f}".format(test_acc))
    print("    Precision:        {0:.4f}".format(test_prec))
    print("    Recall:           {0:.4f}".format(test_rec))
    print("    F1 Score:         {0:.4f}".format(test_f1))
    print("    AUC-ROC:          {0:.4f}".format(test_auc))
    print("  ")
    print("  Best For: Interpretable predictions with clear feature coefficients")
    print("="*100)
    
    print("\n✓✓✓ LOGISTIC REGRESSION MODEL EXECUTION COMPLETE ✓✓✓\n")
    
    return {
        'model': lr_calibrated,
        'scaler': scaler,
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    try:
        results = main()
        print("✓ All operations completed successfully")
    except Exception as e:
        print("ERROR: {0}".format(str(e)))
        import traceback
        traceback.print_exc()
