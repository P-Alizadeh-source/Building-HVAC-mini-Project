"""
Data-driven prediction of next-step indoor temperature
Using Random Forest as a simple and strong baseline
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from simulate import run_baseline_simulation


def create_features(df):
    """
    Create simple features for next-step prediction.
    We predict T_in at time t+1 using information available at time t.
    """
    df = df.copy()

    # Target: next indoor temperature
    df["T_in_next"] = df["T_in"].shift(-1)

    # Features
    df["T_in_lag1"] = df["T_in"]
    df["T_out"] = df["T_out"]
    df["Q_solar"] = df["Q_solar"]
    df["Q_internal"] = df["Q_internal"]
    df["Q_HVAC"] = df["Q_HVAC"]

    # Drop the last row (no next value) and any NaNs
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["T_in_lag1", "T_out", "Q_solar", "Q_internal", "Q_HVAC"]
    X = df[feature_cols]
    y = df["T_in_next"]

    return X, y, feature_cols


def train_and_evaluate(n_steps=168*2):  # 14 days of data
    """
    Generate data, train Random Forest, and evaluate.
    """
    # Generate data with the baseline controller
    df = run_baseline_simulation(n_steps=n_steps)

    X, y, feature_cols = create_features(df)

    # Train/test split (time-series friendly: no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    results = {
        "MAE_train": mean_absolute_error(y_train, y_pred_train),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "model": model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "df": df
    }

    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    print("Training finished.")
    print(f"MAE  (train): {results['MAE_train']:.3f} °C")
    print(f"MAE  (test) : {results['MAE_test']:.3f} °C")
    print(f"RMSE (test) : {results['RMSE_test']:.3f} °C")