# ============================================================================
# PREDICTOR MODULE
# File: src/predictor.py
# ============================================================================
# Makes predictions on new economic data

import numpy as np


def make_predictions(rf_cal, gb_cal, lr_cal, scaler, feature_names):
    """
    Make predictions on sample economic data
    
    Parameters:
    -----------
    rf_cal, gb_cal, lr_cal : calibrated models
        Trained and calibrated models
    scaler : RobustScaler
        Fitted scaler
    feature_names : list
        Names of features
    
    Returns:
    --------
    dict : Dictionary containing predictions and probabilities
    """
    
    print("\n  Creating sample data for healthy economy...")
    
    # Sample data: Healthy economy indicators
    sample_data = np.array([[
        3.5,      # GDP_Growth
        5.5,      # Unemployment
        2.0,      # Inflation
        60.0,     # Debt_to_GDP
        2.5,      # Interest_Rate
        5.0,      # FDI_Change
        2.0,      # Currency_Depreciation
        8.0,      # Credit_Growth
        28.0,     # GDP_Credit_Interaction
        1.5,      # Debt_Interest_Interaction
        10.0      # Currency_FDI_Interaction
    ]])
    
    # Scale sample
    sample_scaled = scaler.transform(sample_data)
    
    # Get predictions
    rf_prob = rf_cal.predict_proba(sample_scaled)[0, 1]
    gb_prob = gb_cal.predict_proba(sample_scaled)[0, 1]
    lr_prob = lr_cal.predict_proba(sample_scaled)[0, 1]
    ensemble_prob = (rf_prob + gb_prob + lr_prob) / 3
    
    print("\n  Sample Prediction Results (Healthy Economy):")
    print(f"    Random Forest:      {rf_prob:.4f} → {'CRISIS' if rf_prob > 0.5 else 'NO CRISIS'}")
    print(f"    Gradient Boosting:  {gb_prob:.4f} → {'CRISIS' if gb_prob > 0.5 else 'NO CRISIS'}")
    print(f"    Logistic Regression:{lr_prob:.4f} → {'CRISIS' if lr_prob > 0.5 else 'NO CRISIS'}")
    print(f"    Ensemble:           {ensemble_prob:.4f} → {'CRISIS' if ensemble_prob > 0.5 else 'NO CRISIS'}")
    
    results = {
        'rf_probability': rf_prob,
        'gb_probability': gb_prob,
        'lr_probability': lr_prob,
        'ensemble_probability': ensemble_prob,
        'sample_data': sample_data
    }
    
    return results


def predict_on_new_data(model, scaler, data_dict):
    """
    Make prediction on new economic data
    
    Parameters:
    -----------
    model : calibrated model
        Trained and calibrated model
    scaler : RobustScaler
        Fitted scaler
    data_dict : dict
        Dictionary with feature names as keys and values as values
    
    Returns:
    --------
    tuple : (prediction, probability)
    """
    
    # Convert dict to array
    feature_values = list(data_dict.values())
    X_new = np.array([feature_values])
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0, 1]
    
    return prediction, probability
