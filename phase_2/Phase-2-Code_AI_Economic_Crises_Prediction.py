

# =============================================================================
# 1. IMPORTS AND SETUP
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

#SHAP support
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

#Setup environment
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

#Create a safe local output folder
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AI FOR ECONOMIC CRISIS PREDICTION - PHASE II")
print("=" * 70)
print(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output Directory: {OUTPUT_DIR}")
print("=" * 70)


# =============================================================================
# 2. DATA GENERATION
# =============================================================================
class EconomicDataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_dataset(self):
        data = {
            'gdp_growth': np.random.normal(2.5, 2.0, self.n_samples),
            'inflation_rate': np.random.normal(2.2, 1.5, self.n_samples),
            'unemployment_rate': np.random.normal(6.5, 2.5, self.n_samples),
            'interest_rate': np.random.normal(3.5, 2.0, self.n_samples),
            'stock_volatility': np.random.exponential(0.25, self.n_samples),
            'current_account_balance': np.random.normal(-2.5, 4.0, self.n_samples),
            'credit_to_gdp_gap': np.random.normal(0, 8, self.n_samples),
            'gov_debt_to_gdp': np.random.normal(65, 25, self.n_samples),
            'commodity_index': np.random.normal(100, 15, self.n_samples)
        }
        df = pd.DataFrame(data)
        prob = 0.1 + np.where(df['gdp_growth'] < 0, 0.3, 0)
        prob += np.where(df['unemployment_rate'] > 10, 0.2, 0)
        prob += np.where(df['inflation_rate'] > 5, 0.15, 0)
        prob += np.where(df['credit_to_gdp_gap'] > 10, 0.15, 0)
        prob = np.clip(prob, 0, 0.8)
        df['crisis'] = np.random.binomial(1, prob)
        return df


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=RANDOM_STATE)

    def preprocess(self, df):
        X, y = df.drop('crisis', axis=1), df['crisis']
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns)
        X_train_res, y_train_res = self.smote.fit_resample(X_train_scaled, y_train)
        return X_train_res, X_test_scaled, y_train_res, y_test, list(X.columns)


# =============================================================================
# 4. MODEL TRAINING
# =============================================================================
class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def train_all(self, X_train, y_train, X_test, y_test):
        self._train_lr(X_train, y_train)
        self._train_rf(X_train, y_train)
        self._train_xgb(X_train, y_train, X_test, y_test)
        return self.models

    def _train_lr(self, X, y):
        model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
        model.fit(X, y)
        self.models['Logistic Regression'] = model

    def _train_rf(self, X, y):
        model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=RANDOM_STATE,
                                       class_weight='balanced', n_jobs=-1)
        model.fit(X, y)
        self.models['Random Forest'] = model

    def _train_xgb(self, X, y, X_test, y_test):
        ratio = (y == 0).sum() / (y == 1).sum()
        model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                  scale_pos_weight=ratio, subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_STATE, eval_metric='auc', n_jobs=-1)
        model.fit(X, y, eval_set=[(X_test, y_test)], verbose=False)
        self.models['XGBoost'] = model

    def evaluate_all(self, X_test, y_test, features):
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }
            self.results[name] = {"metrics": metrics, "confusion_matrix": confusion_matrix(y_test, y_pred)}
            print(f"\n {name} Results:\n", pd.Series(metrics))


# =============================================================================
# 5. VISUALIZATION AND REPORT
# =============================================================================
class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')

    def plot_metrics(self, results):
        df = pd.DataFrame({m: v["metrics"] for m, v in results.items()}).T
        ax = df.plot(kind='bar', figsize=(10, 6), title='Model Performance Comparison', rot=0)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model comparison plot saved at: {save_path}")


def generate_summary_report(results, df, features):
    report_path = os.path.join(OUTPUT_DIR, "phase2_summary_report.txt")
    metrics_df = pd.DataFrame({model: data['metrics'] for model, data in results.items()}).T
    report = f"""
AI FOR ECONOMIC CRISIS PREDICTION - PHASE II SUMMARY REPORT
===========================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Samples: {len(df)}, Features: {len(features)}, Crisis Rate: {df['crisis'].mean():.2%}

MODEL PERFORMANCE
-----------------
{metrics_df.round(4).to_string()}

Best Model: {metrics_df['ROC-AUC'].idxmax()} (ROC-AUC = {metrics_df['ROC-AUC'].max():.4f})
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Summary report saved at: {report_path}")


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    print("\n Starting Phase II Pipeline...")
    df = EconomicDataGenerator().generate_dataset()
    X_train, X_test, y_train, y_test, features = DataPreprocessor().preprocess(df)
    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.evaluate_all(X_test, y_test, features)
    Visualizer().plot_metrics(trainer.results)
    generate_summary_report(trainer.results, df, features)

    print("\n Execution complete. Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
