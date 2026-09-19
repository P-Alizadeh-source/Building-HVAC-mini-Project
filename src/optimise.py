"""
Simple constrained optimisation for HVAC operation
We optimise HVAC power over a short horizon to minimise energy
while keeping temperature inside the comfort band.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from scipy.optimize import minimize
from model import BuildingThermalModel


def optimise_hvac_horizon(
    T_in_current,
    T_out_horizon,
    Q_solar_horizon,
    Q_internal_horizon,
    horizon=6,
    T_min=20.0,
    T_max=24.0,
    Q_max=6000.0,          # maximum heating power
    Q_min=-5000.0,         # maximum cooling power
    comfort_weight=50.0    # how strongly we penalise comfort violations
):
    """
    Optimise HVAC power for the next `horizon` steps.
    
    Returns
    -------
    Q_opt : array of optimal HVAC powers
    """
    model = BuildingThermalModel(dt=3600)

    def objective(Q):
        T = T_in_current
        energy_cost = 0.0
        comfort_penalty = 0.0

        for k in range(horizon):
            T, energy = model.step(
                T,
                T_out_horizon[k],
                Q_internal_horizon[k],
                Q_solar_horizon[k],
                Q[k]
            )
            energy_cost += abs(energy) / 3.6e6   # kWh

            # Soft penalty for being outside comfort band
            if T < T_min:
                comfort_penalty += (T_min - T)**2
            elif T > T_max:
                comfort_penalty += (T - T_max)**2

        return energy_cost + comfort_weight * comfort_penalty

    # Initial guess: do nothing
    Q0 = np.zeros(horizon)

    # Bounds for each time step
    bounds = [(Q_min, Q_max)] * horizon

    result = minimize(
        objective,
        Q0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-4}
    )

    return result.x


def run_optimised_simulation(n_steps=168, horizon=6, T_start=21.0):
    """
    Run a full simulation using the optimised controller
    in a receding-horizon fashion.
    """
    from simulate import generate_weather_and_gains

    model = BuildingThermalModel(dt=3600)
    T_out, Q_solar, Q_internal = generate_weather_and_gains(n_steps)

    T_in = T_start
    records = []

    for k in range(n_steps):
        # Remaining horizon (handle the end of the simulation)
        h = min(horizon, n_steps - k)

        Q_opt = optimise_hvac_horizon(
            T_in_current=T_in,
            T_out_horizon=T_out[k:k+h],
            Q_solar_horizon=Q_solar[k:k+h],
            Q_internal_horizon=Q_internal[k:k+h],
            horizon=h
        )

        # Apply only the first action
        Q_HVAC = Q_opt[0]

        T_next, energy = model.step(
            T_in, T_out[k], Q_internal[k], Q_solar[k], Q_HVAC
        )

        records.append({
            "hour": k,
            "T_out": T_out[k],
            "Q_solar": Q_solar[k],
            "Q_internal": Q_internal[k],
            "Q_HVAC": Q_HVAC,
            "T_in": T_in,
            "energy_J": energy
        })

        T_in = T_next

    import pandas as pd
    df = pd.DataFrame(records)
    df["energy_kWh"] = df["energy_J"] / 3.6e6
    return df
