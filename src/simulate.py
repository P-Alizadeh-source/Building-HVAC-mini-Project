"""
Baseline rule-based HVAC simulation + synthetic data generation
"""

import numpy as np
import pandas as pd
from model import BuildingThermalModel


def generate_weather_and_gains(n_steps=168, dt=3600, seed=42):
    """
    Generate 7 days (168 hours) of synthetic outdoor temperature,
    solar gains and internal gains.
    """
    np.random.seed(seed)
    hours = np.arange(n_steps)

    # Outdoor temperature: daily sine wave + some noise
    T_out = 10 + 8 * np.sin(2 * np.pi * hours / 24 - np.pi/2) + np.random.normal(0, 1.5, n_steps)

    # Solar gains: only during day, simple half-sine
    solar = np.maximum(0, 3000 * np.sin(2 * np.pi * (hours % 24 - 6) / 24))
    solar = np.where((hours % 24 > 6) & (hours % 24 < 18), solar, 0)

    # Internal gains: higher during day (occupancy)
    internal = 400 + 600 * ((hours % 24 > 7) & (hours % 24 < 19)).astype(float)
    internal += np.random.normal(0, 50, n_steps)

    return T_out, solar, internal


def rule_based_controller(T_in, T_min=20.0, T_max=24.0, Q_heat=5000, Q_cool=-4000):
    """
    Simple thermostat:
    - Heat if too cold
    - Cool if too hot
    - Otherwise off
    """
    if T_in < T_min:
        return Q_heat
    elif T_in > T_max:
        return Q_cool
    else:
        return 0.0


def run_baseline_simulation(n_steps=168, T_start=21.0):
    """
    Run the full baseline simulation and return a DataFrame.
    """
    model = BuildingThermalModel(dt=3600)
    T_out, Q_solar, Q_internal = generate_weather_and_gains(n_steps)

    T_in = T_start
    records = []

    for k in range(n_steps):
        Q_HVAC = rule_based_controller(T_in)
        T_next, energy = model.step(T_in, T_out[k], Q_internal[k], Q_solar[k], Q_HVAC)

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

    df = pd.DataFrame(records)
    df["energy_kWh"] = df["energy_J"] / 3.6e6   # convert J → kWh
    return df


# -------------------------------------------------
# Quick test when running this file directly
# -------------------------------------------------
if __name__ == "__main__":
    df = run_baseline_simulation()

    print("Simulation finished.")
    print(f"Total HVAC energy: {df['energy_kWh'].sum():.1f} kWh")
    print(f"Average indoor temperature: {df['T_in'].mean():.2f} °C")
    print(f"Min / Max indoor temperature: {df['T_in'].min():.1f} / {df['T_in'].max():.1f} °C")
    print("\nFirst 5 rows:")
    print(df.head())