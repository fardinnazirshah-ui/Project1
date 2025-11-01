# ============================================================================
# DATA GENERATOR MODULE
# File: src/data_generator.py
# ============================================================================
# Generates synthetic economic data with realistic crisis patterns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def generate_economic_data(n_countries=8, n_years=27):
    """
    Generate synthetic historical economic data with crisis patterns
    
    Parameters:
    -----------
    n_countries : int
        Number of countries to simulate
    n_years : int
        Number of years of quarterly data
    
    Returns:
    --------
    tuple : (df, X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    
    # Define historical crisis periods
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
                date_str = f"{year}-{quarter}"
                is_crisis = 1 if date_str in crisis_periods.get(country, []) else 0
                
                # Generate data based on crisis status
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
                
                # Engineered features
                gdp_credit_interaction = gdp_growth * credit_growth
                debt_interest_interaction = (debt_to_gdp / 100) * interest_rate
                currency_fdi_interaction = abs(currency_depreciation) * abs(fdi_change)
                
                record = {
                    'GDP_Growth': round(gdp_growth, 2),
                    'Unemployment': round(unemployment, 2),
                    'Inflation': round(inflation, 2),
                    'Debt_to_GDP': round(debt_to_gdp, 2),
                    'Interest_Rate': round(interest_rate, 2),
                    'FDI_Change': round(fdi_change, 2),
                    'Currency_Depreciation': round(currency_depreciation, 2),
                    'Credit_Growth': round(credit_growth, 2),
                    'GDP_Credit_Interaction': round(gdp_credit_interaction, 2),
                    'Debt_Interest_Interaction': round(debt_interest_interaction, 2),
                    'Currency_FDI_Interaction': round(currency_fdi_interaction, 2),
                    'Crisis': is_crisis
                }
                
                data_records.append(record)
    
    df = pd.DataFrame(data_records)
    
    # Separate features and target
    X = df.drop('Crisis', axis=1).values
    y = df['Crisis'].values
    feature_names = df.drop('Crisis', axis=1).columns.tolist()
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
