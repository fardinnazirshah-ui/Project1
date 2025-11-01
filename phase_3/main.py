# ============================================================================
# ECONOMIC CRISIS PREDICTION MODEL - MAIN ENTRY POINT (FIXED VERSION)
# File: src/main.py
# ============================================================================
# FIXED: Proper import handling for local modules
# This is the corrected version that will work in PyCharm

"""
Economic Crisis Prediction System - FIXED VERSION
Author: Fardin
Date: November 1, 2025
Version: 3.0 - FIXED
"""

import sys
import os

# FIXED: Add current directory to Python path BEFORE importing
# This ensures Python imports from local files, not site-packages
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import local modules
try:
    from data_generator import generate_economic_data
    from model_trainer import train_models
    from evaluator import evaluate_models
    from predictor import make_predictions
    print("✓ All local modules imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import modules: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function
    Orchestrates the entire pipeline:
    1. Generate data
    2. Train models
    3. Evaluate models
    4. Make predictions
    """
    
    print("="*100)
    print(" "*25 + "ECONOMIC CRISIS PREDICTION MODEL - FIXED VERSION")
    print(" "*35 + "Version 3.0 - Now Working!")
    print("="*100)
    
    try:
        # Step 1: Generate Data
        print("\n[STEP 1] Generating Economic Data...")
        df, X_train, X_test, y_train, y_test, scaler, feature_names = generate_economic_data()
        logger.info(f"Generated {len(df)} samples with {y_train.sum()} crises in training set")
        print(f"✓ Generated {len(df)} samples")
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        
        # Step 2: Train Models
        print("\n[STEP 2] Training Models...")
        rf_model, gb_model, lr_model, rf_cal, gb_cal, lr_cal = train_models(
            X_train, X_test, y_train, y_test, scaler
        )
        logger.info("Models trained and calibrated successfully")
        print("✓ All models trained and calibrated")
        
        # Step 3: Evaluate Models
        print("\n[STEP 3] Evaluating Models...")
        results = evaluate_models(
            rf_cal, gb_cal, lr_cal, X_test, y_test, scaler
        )
        logger.info("Model evaluation complete")
        print("✓ Evaluation complete")
        
        # Step 4: Make Predictions
        print("\n[STEP 4] Making Predictions...")
        predictions = make_predictions(
            rf_cal, gb_cal, lr_cal, scaler, feature_names
        )
        logger.info("Predictions generated")
        print("✓ Predictions generated")
        
        # Print summary
        print_summary(results)
        
        print("\n" + "="*100)
        print("✓✓✓ EXECUTION COMPLETE - NO ERRORS ✓✓✓")
        print("="*100)
        
        return results
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(results):
    """Print execution summary"""
    print("\n" + "-"*100)
    print("SUMMARY")
    print("-"*100)
    
    if results:
        print("\nModel Performance:")
        print(f"  Random Forest AUC:        {results['rf_auc']:.4f}")
        print(f"  Gradient Boosting AUC:    {results['gb_auc']:.4f}")
        print(f"  Logistic Regression AUC:  {results['lr_auc']:.4f}")
        print(f"  Ensemble AUC:             {results['ensemble_auc']:.4f}")
        
        print("\nBest Performance:")
        best_auc = max(results['rf_auc'], results['gb_auc'], results['lr_auc'])
        if best_auc == results['rf_auc']:
            print("  ✓ Random Forest - Best overall AUC")
        elif best_auc == results['gb_auc']:
            print("  ✓ Gradient Boosting - Best overall AUC")
        else:
            print("  ✓ Logistic Regression - Best overall AUC")


if __name__ == "__main__":
    # Run main program
    main()
