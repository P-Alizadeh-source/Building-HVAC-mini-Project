"""
Data-driven predictive HVAC control.

The Random Forest predicts the next indoor temperature.
The optimizer searches for HVAC actions that balance
electrical energy use and thermal comfort.
"""

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from hvac import HVACSystem
from model import BuildingThermalModel
from predict import train_and_evaluate
from simulate import generate_weather_and_gains


def predict_next_temperature(
    model,
    T_in,
    T_mass,
    T_out,
    Q_solar,
    Q_internal,
    Q_HVAC,
):
    """
    Predict next-step indoor temperature
    using the trained ML model.
    """

    X = pd.DataFrame([{
        "T_in": T_in,
        "T_mass": T_mass,
        "T_out": T_out,
        "Q_solar": Q_solar,
        "Q_internal": Q_internal,
        "Q_HVAC": Q_HVAC,
    }])

    return model.predict(X)[0]


def optimise_hvac_action(
    rf_model,
    T_in,
    T_mass,
    T_out_horizon,
    Q_solar_horizon,
    Q_internal_horizon,
    horizon=6,
    T_min=20.0,
    T_max=24.0,
    Q_min=-5000.0,
    Q_max=6000.0,
    energy_weight=1.0,
    comfort_weight=20.0,
):
    """
    Optimize HVAC power using the ML prediction model.
    """

    hvac = HVACSystem()

    def objective(Q_sequence):

        T = T_in
        T_mass_current = T_mass

        total_cost = 0.0

        for k in range(horizon):

            Q = Q_sequence[k]

            T_next = predict_next_temperature(
                rf_model,
                T,
                T_mass_current,
                T_out_horizon[k],
                Q_solar_horizon[k],
                Q_internal_horizon[k],
                Q,
            )

            P_electric = (
                hvac.electrical_power(Q)
            )

            energy_kWh = (
                P_electric * 3600 / 3.6e6
            )

            # Soft comfort penalty
            lower_violation = max(
                0.0,
                T_min - T_next
            )

            upper_violation = max(
                0.0,
                T_next - T_max
            )

            comfort_penalty = (
                lower_violation ** 2
                + upper_violation ** 2
            )

            total_cost += (
                energy_weight * energy_kWh
                + comfort_weight
                * comfort_penalty
            )

            T = T_next

            # For the ML model, we keep the mass
            # state approximately unchanged here.
            # We will improve this later if needed.

        return total_cost

    Q0 = np.zeros(horizon)

    bounds = [
        (Q_min, Q_max)
        for _ in range(horizon)
    ]

    result = minimize(
        objective,
        Q0,
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 200,
            "ftol": 1e-5,
        },
    )

    return result.x


def run_data_driven_control(
    n_steps=168,
    horizon=6,
    T_start=21.0,
):
    """
    Run the complete data-driven MPC simulation.
    """

    # Train the ML model
    prediction_output = (
        train_and_evaluate()
    )

    rf_model = (
        prediction_output["random_forest"]
    )

    building = BuildingThermalModel()

    T_out, Q_solar, Q_internal = (
        generate_weather_and_gains(
            n_steps
        )
    )

    T_in = T_start
    T_mass = T_start

    records = []

    for k in range(n_steps):

        h = min(
            horizon,
            n_steps - k
        )

        Q_opt = optimise_hvac_action(
            rf_model=rf_model,
            T_in=T_in,
            T_mass=T_mass,
            T_out_horizon=T_out[k:k+h],
            Q_solar_horizon=Q_solar[k:k+h],
            Q_internal_horizon=Q_internal[k:k+h],
            horizon=h,
        )

        Q_HVAC = Q_opt[0]

        T_next, T_mass_next = (
            building.step(
                T_air=T_in,
                T_mass=T_mass,
                T_out=T_out[k],
                Q_internal=Q_internal[k],
                Q_solar=Q_solar[k],
                Q_HVAC=Q_HVAC,
            )
        )

        records.append({
            "hour": k,
            "T_out": T_out[k],
            "Q_solar": Q_solar[k],
            "Q_internal": Q_internal[k],
            "Q_HVAC": Q_HVAC,
            "T_in": T_in,
            "T_mass": T_mass,
        })

        T_in = T_next
        T_mass = T_mass_next

    df = pd.DataFrame(records)

    hvac = HVACSystem()

    df["P_electric"] = df[
        "Q_HVAC"
    ].apply(
        hvac.electrical_power
    )

    df["energy_kWh"] = (
        df["P_electric"] * 3600 / 3.6e6
    )

    return df


if __name__ == "__main__":

    df = run_data_driven_control()

    print("\nData-driven control")
    print("=" * 50)

    print(
        f"Electrical energy: "
        f"{df['energy_kWh'].sum():.2f} kWh"
    )

    print(
        f"Indoor temperature range: "
        f"{df['T_in'].min():.2f} - "
        f"{df['T_in'].max():.2f} °C"
    )