"""
In this file we'll compare:
- SR model with and without normalization of the features
- SR model with different transformations of the target variable (log, sqrt, boxcox)
- SR with the best configuration VS other ML models (Random Forest, XGBoost, etc.)

--- Summary of Results ---
Normalization Results: No-Norm — R2: 0.3055, MAE: 4.1803; Norm — R2: 0.4205, MAE: 3.8525

Transformation Results: log — R2: 0.4278, MAE: 3.7072; sqrt — R2: 0.4280, MAE: 3.7737; boxcox — R2: 0.2944, MAE: 4.1307

ML Model Results: 
{'Base SR Model': {'R2': 0.28723610079914685, 'MAE': 4.349527427720096}, 
'Linear Regression': {'R2': 0.3753256957577681, 'MAE': 3.9979428773874277}, 
'Random Forest': {'R2': 0.8388628759315451, 'MAE': 1.8692855873015806}, 
'XGBoost': {'R2': 0.801423386606432, 'MAE': 2.091650475299169}}
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pysr import PySRRegressor

# Import all your custom data preparation functions
from data_prep import * 


def evaluate_normalization(X, y):
    # Evaluate the effect of normalization on the features
    train_df_no_norm, test_df_no_norm = split_data(X, y)

    train_X_no_norm = train_df_no_norm.drop(columns='scaled-sound-pressure')
    test_X_no_norm = test_df_no_norm.drop(columns='scaled-sound-pressure')
    train_y = train_df_no_norm['scaled-sound-pressure']
    test_y = test_df_no_norm['scaled-sound-pressure']

    train_X_norm = normalize_features(train_X_no_norm)
    test_X_norm = normalize_features(test_X_no_norm, reference=train_X_no_norm)
    
    # Sanitize feature names for PySR and train/evaluate on normalized data
    train_X_norm_s, test_X_norm_s, _ = sanitize_pysr_columns(train_X_norm, test_X_norm)
    model_norm = train_sr_model(train_X_norm_s, train_y)
    r2_norm, mae_norm = evaluate_model(model_norm, test_X_norm_s, test_y)

    # Sanitize feature names for PySR and train/evaluate on non-normalized data
    train_X_no_norm_s, test_X_no_norm_s, _ = sanitize_pysr_columns(train_X_no_norm, test_X_no_norm)
    model_no_norm = train_sr_model(train_X_no_norm_s, train_y)
    r2_no_norm, mae_no_norm = evaluate_model(model_no_norm, test_X_no_norm_s, test_y)

    print(f"Normalized Features - R2: {r2_norm:.4f}, MAE: {mae_norm:.4f}")
    print(f"Non-Normalized Features - R2: {r2_no_norm:.4f}, MAE: {mae_no_norm:.4f}")
    return (r2_no_norm, mae_no_norm), (r2_norm, mae_norm)

def evaluate_transformation(X, y, transformation):
    # Evaluate the effect of different transformations on the target variable
    if transformation == 'log':
        y_transformed = np.log1p(y)
    elif transformation == 'sqrt':
        y_transformed = np.sqrt(y)
    elif transformation == 'boxcox':
        # Need to save lambda to inverse transform later
        y_transformed, lmbda = stats.boxcox(y + 1) 
    else:
        raise ValueError("Unsupported transformation. Choose from 'log', 'sqrt', or 'boxcox'.")

    # Split the data
    train_df, test_df = split_data(X, pd.Series(y_transformed, name='scaled-sound-pressure'))

    # Prepare feature DataFrames and sanitize column names for PySR
    X_train = train_df.drop(columns='scaled-sound-pressure')
    X_test = test_df.drop(columns='scaled-sound-pressure')
    y_train = train_df['scaled-sound-pressure']
    y_test_transformed = test_df['scaled-sound-pressure']

    X_train_s, X_test_s, _ = sanitize_pysr_columns(X_train, X_test)

    # Train model
    model = train_sr_model(X_train_s, y_train)

    # Generate predictions on the test set
    predictions_transformed = model.predict(X_test_s)
    
    # Inverse transform predictions and ground truth to compare fairly
    if transformation == 'log':
        predictions_original = np.expm1(predictions_transformed)
        y_test_original = np.expm1(y_test_transformed)
    elif transformation == 'sqrt':
        predictions_original = np.square(predictions_transformed)
        y_test_original = np.square(y_test_transformed)
    elif transformation == 'boxcox':
        predictions_original = inv_boxcox(predictions_transformed, lmbda) - 1
        y_test_original = inv_boxcox(y_test_transformed, lmbda) - 1

    # Calculate metrics
    r2 = r2_score(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    
    print(f"Transformation: {transformation} - R2: {r2:.4f}, MAE: {mae:.4f}")
    return r2, mae


def evaluate_models(X, y, models):
    # Evaluate different ML models (Random Forest, XGBoost, etc.) against the SR model
    results = {}
    train_df, test_df = split_data(X, y)
    
    for name, model in models.items():
        # Fit the model. Use train_sr_model for PySRRegressor so we sanitize
        # columns and suppress any on-disk outputs; other models use their
        # standard fit/predict interface.
        X_train = train_df.drop(columns='scaled-sound-pressure')
        y_train = train_df['scaled-sound-pressure']
        X_test = test_df.drop(columns='scaled-sound-pressure')

        if isinstance(model, PySRRegressor):
            # For PySR, drop rows with NaNs (PySR doesn't accept NaNs),
            # sanitize both train and test column names, then train/predict.
            train_mask = X_train.notna().all(axis=1) & y_train.notna()
            X_train_clean = X_train.loc[train_mask]
            y_train_clean = y_train.loc[train_mask]

            # align test X and y
            y_test = test_df['scaled-sound-pressure']
            test_mask = X_test.notna().all(axis=1) & y_test.notna()
            X_test_clean = X_test.loc[test_mask]
            y_test_clean = y_test.loc[test_mask]

            # If nothing remains after dropping NaNs, skip this model
            if len(X_train_clean) == 0 or len(X_test_clean) == 0:
                print(f"Skipping {name}: no valid rows after dropping NaNs for PySR.")
                preds = np.array([])
                y_test_used = np.array([])
            else:
                X_train_s, X_test_s, _ = sanitize_pysr_columns(X_train_clean, X_test_clean)
                fitted = train_sr_model(X_train_s, y_train_clean)
                preds = fitted.predict(X_test_s)
                y_test_used = y_test_clean
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            y_test_used = test_df['scaled-sound-pressure']
        
        # If predictions are empty (skipped), record NaNs
        if preds.size == 0:
            r2 = float('nan')
            mae = float('nan')
        else:
            r2 = r2_score(y_test_used, preds)
            mae = mean_absolute_error(y_test_used, preds)
        
        results[name] = {'R2': r2, 'MAE': mae}
        print(f"Model: {name} - R2: {r2:.4f}, MAE: {mae:.4f}")
        
    return results


if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    results = {}
    print("--- Fetching NASA Airfoil Dataset ---")
    dataset = fetch_ucirepo(id=291) 
    X = dataset.data.features 
    y = dataset.data.targets.squeeze()
    
    print("\n--- 1. Evaluating Normalization ---")
    normalization_results = evaluate_normalization(X, y)
    
    print("\n--- 2. Evaluating Transformations ---")
    log_results = evaluate_transformation(X, y, 'log')
    sqrt_results = evaluate_transformation(X, y, 'sqrt')
    boxcox_results = evaluate_transformation(X, y, 'boxcox')

    print("--- 3. Benchmarking Traditional ML Models ---")
    # Only benchmark classical ML models here; avoid fitting PySR in this loop
    models_to_test = {
        'Base SR Model':PySRRegressor(
            niterations=1000,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "sqrt", "square", "exp"],
            populations=20,             # Number of populations in the genetic algorithm
            maxsize=40,                 # Max complexity
            model_selection="best",     
            loss="loss(x, y) = (x - y)^2", # Use mean squared error as the loss function
            random_state=42,
            procs=4,                    
            verbosity=1,
            parallelism='serial',
            deterministic=True
        ),  
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    model_results = evaluate_models(X, y, models_to_test)
    
    results['normalization'] = normalization_results
    results['transformations'] = {
        'log': log_results,
        'sqrt': sqrt_results,
        'boxcox': boxcox_results
    }
    results['ml_models'] = model_results

    print("\n--- Summary of Results ---")
    no_norm = results['normalization'][0]
    norm = results['normalization'][1]
    print(f"Normalization Results: No-Norm — R2: {no_norm[0]:.4f}, MAE: {no_norm[1]:.4f}; Norm — R2: {norm[0]:.4f}, MAE: {norm[1]:.4f}")
    tr = results['transformations']
    print(f"Transformation Results: log — R2: {tr['log'][0]:.4f}, MAE: {tr['log'][1]:.4f}; sqrt — R2: {tr['sqrt'][0]:.4f}, MAE: {tr['sqrt'][1]:.4f}; boxcox — R2: {tr['boxcox'][0]:.4f}, MAE: {tr['boxcox'][1]:.4f}")
    print(f"ML Model Results: {results['ml_models']}")