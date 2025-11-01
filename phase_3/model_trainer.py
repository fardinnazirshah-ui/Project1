# ============================================================================
# MODEL TRAINER MODULE
# File: src/model_trainer.py
# ============================================================================
# Trains and calibrates machine learning models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def train_models(X_train, X_test, y_train, y_test, scaler):
    """
    Train three complementary models with optimized hyperparameters
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features (scaled)
    y_train, y_test : array-like
        Training and test targets
    scaler : RobustScaler
        Fitted scaler object
    
    Returns:
    --------
    tuple : (rf_model, gb_model, lr_model, rf_calibrated, gb_calibrated, lr_calibrated)
    """
    
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.5,
        class_weight={0: 1, 1: 10},
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("  ✓ Random Forest trained")
    
    print("  Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    print("  ✓ Gradient Boosting trained")
    
    print("  Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    lr_model.fit(X_train, y_train)
    print("  ✓ Logistic Regression trained")
    
    print("\n  Applying Probability Calibration...")
    rf_calibrated = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
    rf_calibrated.fit(X_train, y_train)
    
    gb_calibrated = CalibratedClassifierCV(gb_model, method='sigmoid', cv=5)
    gb_calibrated.fit(X_train, y_train)
    
    lr_calibrated = CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
    lr_calibrated.fit(X_train, y_train)
    print("  ✓ Probability calibration complete")
    
    return rf_model, gb_model, lr_model, rf_calibrated, gb_calibrated, lr_calibrated
