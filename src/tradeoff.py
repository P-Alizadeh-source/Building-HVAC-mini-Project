import numpy as np


def calculate_objective(
    energy,
    temperatures,
    setpoint=22.0,
    energy_weight=1.0,
    comfort_weight=10.0,
):
    temperatures = np.array(temperatures)

    comfort_penalty = np.mean(
        (temperatures - setpoint) ** 2
    )

    objective = (
        energy_weight * energy
        + comfort_weight * comfort_penalty
    )

    return objective