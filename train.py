"""
The results from this script may differ from the results in the notebook due to the stochastic nature of PySR.
But the metrics should be similar. The final model is trained on the entire dataset, and the best equation is saved to a text file.
"""

import numpy as np
import pandas as pd
import pickle
from pysr import PySRRegressor
from data_prep import load_data, split_data, sanitize_pysr_columns

def main():
    dataset, X, y = load_data()
    
    # Best config: Log Transformation on Target (No normalization needed)
    y_log = np.log1p(y.squeeze())
    
    # Split data safely
    train_df, test_df = split_data(X, pd.Series(y_log, name='scaled-sound-pressure'))
    X_train = train_df.drop(columns='scaled-sound-pressure')
    y_train = train_df['scaled-sound-pressure']
    
    # Sanitize for PySR
    X_train_s, _, _ = sanitize_pysr_columns(X_train)
    
    # Define final model
    print("Training final PySR model...")
    model = PySRRegressor(
        niterations=1000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "square", "exp"],
        populations=30,             
        maxsize=30,                 
        model_selection="best",     
        loss="loss(x, y) = (x - y)^2", 
        random_state=42,
        parallelism='serial', 
        deterministic=True # we want reproducibility for the final model
    )
    
    model.fit(X_train_s, y_train)

    # Althoug PySR can save the equation and some outputs, i prefer to save the model and equation explicitly
    # Save the final equation string to a text file
    with open("final_equation.txt", "w") as f:
        f.write(str(model.sympy()))
        
    # Save the model state using pickle
    with open("final_sr_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Training complete. Model and equation saved.")

if __name__ == "__main__":
    main()