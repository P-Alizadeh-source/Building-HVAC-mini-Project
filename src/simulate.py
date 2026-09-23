"""
Building simulation and synthetic weather/load generation.
"""

import numpy as np
import pandas as pd

from model import BuildingThermalModel
from hvac import HVACSystem


def generate_weather_and_gains(
    n_steps=168,
    dt=3600,
    seed=42
):
    """
    Generate synthetic weather and building heat gains.

    Returns
    -------
    T_out : ndarray
        Outdoor temperature [°C]

    Q_solar : ndarray
        Solar gains [W]

    Q_internal : ndarray
        Internal gains from occupants/equipment [W]
    """

    rng = np.random.default_rng(seed)

    hours = np.arange(n_steps)

    # Outdoor temperature
    T_out = (
        10
        + 8 * np.sin(
            2 * np.pi * hours / 24 - np.pi / 2
        )
        + rng.normal(0, 1.5, n_steps)
    )

    # Solar gains
    solar = np.maximum(
        0,
        3000
        * np.sin(
            2 * np.pi * (hours % 24 - 6) / 24
        )
    )

    solar = np.where(
        (hours % 24 > 6)
        & (hours % 24 < 18),
        solar,
        0
    )

    # Internal gains
    occupied = (
        (hours % 24 > 7)
        & (hours % 24 < 19)
    )

    internal = (
        400
        + 600 * occupied.astype(float)
        + rng.normal(0, 50, n_steps)
    )

    return T_out, solar, internal


def rule_based_controller(
    T_air,
    T_min=20.0,
    T_max=24.0,
    Q_heat=5000.0,
    Q_cool=-4000.0
):
    """
    Simple thermostat controller.
    """

    if T_air < T_min:
        return Q_heat

    elif T_air > T_max:
        return Q_cool

    return 0.0


def run_baseline_simulation(
    n_steps=168,
    T_start=21.0,
    T_mass_start=21.0
):
    """
    Run baseline thermostat simulation.
    """

    building = BuildingThermalModel()
    hvac = HVACSystem()

    T_out, Q_solar, Q_internal = (
        generate_weather_and_gains(n_steps)
    )

    T_air = T_start
    T_mass = T_mass_start

    records = []

    for k in range(n_steps):

        Q_HVAC = rule_based_controller(T_air)

        P_electric = hvac.electrical_power(Q_HVAC)

        T_air_next, T_mass_next = building.step(
            T_air=T_air,
            T_mass=T_mass,
            T_out=T_out[k],
            Q_internal=Q_internal[k],
            Q_solar=Q_solar[k],
            Q_HVAC=Q_HVAC,
        )

        records.append({
            "hour": k,
            "T_out": T_out[k],
            "Q_solar": Q_solar[k],
            "Q_internal": Q_internal[k],
            "Q_HVAC": Q_HVAC,
            "P_electric": P_electric,
            "T_in": T_air,
            "T_mass": T_mass,
        })

        T_air = T_air_next
        T_mass = T_mass_next

    df = pd.DataFrame(records)

    df["energy_kWh"] = (
        df["P_electric"] * 3600 / 3.6e6
    )

    return df


if __name__ == "__main__":

    df = run_baseline_simulation()

    print("Simulation finished.")

    print(
        f"Total electrical energy: "
        f"{df['energy_kWh'].sum():.2f} kWh"
    )

    print(
        f"Indoor temperature range: "
        f"{df['T_in'].min():.2f} – "
        f"{df['T_in'].max():.2f} °C"
    )