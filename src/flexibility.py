import numpy as np


def generate_tou_price(hours):
    """
    Generate a simple time-of-use electricity price profile.

    Prices are illustrative and are not real electricity tariffs.
    """

    prices = np.zeros(hours)

    for hour in range(hours):

        hour_of_day = hour % 24

        if 8 <= hour_of_day < 12:
            prices[hour] = 0.25

        elif 17 <= hour_of_day < 21:
            prices[hour] = 0.30

        else:
            prices[hour] = 0.12

    return prices


def calculate_energy_cost(
    power_kw,
    prices,
    timestep_hours=1.0,
):
    """
    Calculate electricity cost.
    """

    power_kw = np.asarray(power_kw)
    prices = np.asarray(prices)

    energy_kwh = power_kw * timestep_hours

    cost = np.sum(
        energy_kwh * prices
    )

    return cost


def calculate_peak_power(power_kw):
    """
    Calculate maximum electrical power demand.
    """

    power_kw = np.asarray(power_kw)

    return np.max(power_kw)