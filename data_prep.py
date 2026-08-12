from ucimlrepo import fetch_ucirepo 
import sklearn as sk
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from pysr import PySRRegressor
import re


GROUP_COLS = ["attack-angle", "chord-length", "free-stream-velocity"]

def sanitize_pysr_columns(X_train, X_test=None, ignore_columns=None):
    """
    Rename DataFrame columns to valid PySR variable names (alphanumeric and underscores).
    Returns renamed copies of X_train and X_test (if provided) and the mapping dict.
    """
    if ignore_columns is None:
        ignore_columns = []

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train).copy()
    else:
        X_train = X_train.copy()

    mapping = {}
    new_names = []
    seen = {}
    for col in X_train.columns:
        if col in ignore_columns:
            new = col
        else:
            new = re.sub(r"[^0-9a-zA-Z_]", "_", str(col))
            # ensure it doesn't start with a number
            if re.match(r"^[0-9]", new):
                new = f"_{new}"
            # avoid duplicates
            if new in seen:
                seen[new] += 1
                new = f"{new}_{seen[new]}"
            else:
                seen[new] = 0
        mapping[col] = new
        new_names.append(new)

    X_train.columns = new_names

    X_test_out = None
    if X_test is not None:
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test).copy()
        else:
            X_test = X_test.copy()

        # apply same mapping, add any missing columns if necessary
        test_new = []
        for col in X_test.columns:
            if col in mapping:
                test_new.append(mapping[col])
            else:
                if col in ignore_columns:
                    test_new.append(col)
                else:
                    new = re.sub(r"[^0-9a-zA-Z_]", "_", str(col))
                    if re.match(r"^[0-9]", new):
                        new = f"_{new}"
                    # ensure uniqueness relative to train mapping
                    if new in seen:
                        seen[new] += 1
                        new = f"{new}_{seen[new]}"
                    else:
                        seen[new] = 0
                    mapping[col] = new
                    test_new.append(new)

        X_test.columns = test_new
        X_test_out = X_test

    return X_train, X_test_out, mapping



def load_data():
    # Obtain the dataset from the UCI Machine Learning Repository
    dataset = fetch_ucirepo(id=291) 
    
    # data (as pandas dataframes) 
    X = dataset.data.features 
    y = dataset.data.targets 

    return dataset, X,y

def normalize_features(X, reference=None):
    # Add normalization to the features to avoid frequency bias (tens of thousands of Hz vs. tens of deg)
    X_df = pd.DataFrame(X).copy()
    reference_df = X_df if reference is None else pd.DataFrame(reference).reindex(columns=X_df.columns)

    norms = np.linalg.norm(reference_df.to_numpy(dtype=float), axis=0)
    norms = np.where(norms == 0, 1.0, norms)

    X_norm = X_df.astype(float).div(norms, axis=1)
    return X_norm

def log_transform_target(y):
    # Apply log transformation to the target variable
    y_loh = np.log1p(y)  
    return y_loh

def sqrt_transform_target(y):
    # Apply square root transformation to the target variable
    y_sqrt = np.sqrt(y)  
    return y_sqrt

def boxcox_transform_target(y):
    # Apply Box-Cox transformation to the target variable
    y_box_cox, _ = stats.boxcox(y + 1)  
    return y_box_cox

def split_data(X, y):
    df = X.copy()
    df['scaled-sound-pressure'] = y.values

    # Grouping by physical experiment identifiers to avoid data leakage
    missing_group_cols = [col for col in GROUP_COLS if col not in df.columns]
    if missing_group_cols:
        raise ValueError(
            "split_data requires the raw feature columns used for grouping: "
            f"{missing_group_cols}"
        )

    df["group_id"] = df[GROUP_COLS].astype(str).agg("-".join, axis=1)

    print(f"Unique groups: {df['group_id'].nunique()}")
    print(f"Total rows: {len(df)}")
    print(f"Average rows per group: {len(df) / df['group_id'].nunique():.1f}")

    # Split by group, not by row to avoid data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["group_id"]))

    train_df = df.iloc[train_idx].drop(columns="group_id")
    test_df = df.iloc[test_idx].drop(columns="group_id")

    # Check if any group appears in both splits
    train_groups = set(df.iloc[train_idx]["group_id"])
    test_groups = set(df.iloc[test_idx]["group_id"])
    assert len(train_groups & test_groups) == 0, "there's a leakage"

    print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows")
    print("No group overlap detected.")

    return train_df, test_df

def train_sr_model(X_train, y_train, niterations=1000, populations=20, maxsize=20):
    # Train the Symbolic Regression model

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt"],
        populations=populations,
        maxsize=maxsize,
        loss="loss(x, y) = (x - y)^2",
        verbosity=0,
        random_state=42,
    )

    # PySR requires valid variable names for DataFrame columns. If a DataFrame is
    # provided, sanitize column names in place before fitting.
    if isinstance(X_train, pd.DataFrame):
        X_train, _, _ = sanitize_pysr_columns(X_train)

    # Perform a standard fit and leave PySR outputs on disk (basic training)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R**2 test: {r2:.4f}")
    print(f"MAE test: {mae:.4f}")

    return r2, mae