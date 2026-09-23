"""
Data-driven next-step indoor temperature prediction.

Models:
1. Persistence baseline
2. Linear Regression
3. Random Forest
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from simulate import run_baseline_simulation


def create_features(df):
    """
    Create features for one-step-ahead prediction.
    """

    df = df.copy()

    df["T_in_next"] = df["T_in"].shift(-1)

    feature_cols = [
        "T_in",
        "T_mass",
        "T_out",
        "Q_solar",
        "Q_internal",
        "Q_HVAC",
    ]

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df["T_in_next"]

    return X, y, feature_cols


def persistence_prediction(X):
    """
    Persistence model:
    T(t+1) = T(t)
    """

    return X["T_in"].values


def evaluate_predictions(y_true, y_pred):
    """
    Calculate standard regression metrics.
    """

    return {
        "MAE": mean_absolute_error(
            y_true, y_pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true, y_pred
            )
        ),
        "R2": r2_score(
            y_true, y_pred
        ),
    }


def train_and_evaluate(
    n_steps=24 * 30
):
    """
    Train and compare prediction models.
    """

    df = run_baseline_simulation(
        n_steps=n_steps
    )

    X, y, feature_cols = create_features(df)

    # Time-series split
    split_1 = int(len(X) * 0.70)
    split_2 = int(len(X) * 0.85)

    X_train = X.iloc[:split_1]
    y_train = y.iloc[:split_1]

    X_val = X.iloc[
        split_1:split_2
    ]
    y_val = y.iloc[
        split_1:split_2
    ]

    X_test = X.iloc[split_2:]
    y_test = y.iloc[split_2:]

    results = {}

    # ------------------------------------------------
    # 1. Persistence
    # ------------------------------------------------

    y_pred_persistence = persistence_prediction(
        X_test
    )

    results["Persistence"] = evaluate_predictions(
        y_test,
        y_pred_persistence
    )

    # ------------------------------------------------
    # 2. Linear Regression
    # ------------------------------------------------

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    y_pred_linear = linear_model.predict(
        X_test
    )

    results["Linear Regression"] = (
        evaluate_predictions(
            y_test,
            y_pred_linear
        )
    )

    # ------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=1,
    )

    rf_model.fit(
        X_train,
        y_train
    )

    y_pred_rf = rf_model.predict(
        X_test
    )

    results["Random Forest"] = (
        evaluate_predictions(
            y_test,
            y_pred_rf
        )
    )

    return {
        "results": results,
        "random_forest": rf_model,
        "linear_model": linear_model,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_rf": y_pred_rf,
    }


if __name__ == "__main__":

    output = train_and_evaluate()

    print("\nPrediction results")
    print("=" * 50)

    for model_name, metrics in (
        output["results"].items()
    ):

        print(f"\n{model_name}")

        print(
            f"MAE  : {metrics['MAE']:.3f} °C"
        )

        print(
            f"RMSE : {metrics['RMSE']:.3f} °C"
        )

        print(
            f"R²   : {metrics['R2']:.3f}"
        )