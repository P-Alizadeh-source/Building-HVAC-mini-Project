import numpy as np


def calculate_prediction_metrics(actual, predicted):
    """
    Calculate prediction accuracy metrics.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    max_error = np.max(
        np.abs(actual - predicted)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error,
    }


def calculate_total_energy(energy_values):
    """
    Calculate total electrical energy consumption.
    """

    energy_values = np.asarray(energy_values)

    return np.sum(energy_values)


def calculate_comfort_metrics(
    temperatures,
    lower_limit=20.0,
    upper_limit=24.0,
):
    """
    Calculate thermal comfort performance.
    """

    temperatures = np.asarray(temperatures)

    uncomfortable = (
        (temperatures < lower_limit)
        | (temperatures > upper_limit)
    )

    discomfort_hours = np.sum(uncomfortable)

    comfort_percentage = (
        100
        * (1 - discomfort_hours / len(temperatures))
    )

    return {
        "discomfort_hours": discomfort_hours,
        "comfort_percentage": comfort_percentage,
    }


def calculate_peak_power(power_values):
    """
    Calculate maximum electrical power demand.
    """

    power_values = np.asarray(power_values)

    return np.max(power_values)