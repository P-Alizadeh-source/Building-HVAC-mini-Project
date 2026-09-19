"""
Simple first-order building thermal model
C * dT_in/dt = Q_internal + Q_solar + Q_HVAC - UA * (T_in - T_out)
"""

import numpy as np


class BuildingThermalModel:
    def __init__(self, C=5e6, UA=200, dt=3600):
        """
        Parameters
        ----------
        C : float
            Thermal capacity of the building [J/K]
        UA : float
            Overall heat loss coefficient [W/K]
        dt : float
            Time step [seconds] (default = 1 hour)
        """
        self.C = C
        self.UA = UA
        self.dt = dt

    def step(self, T_in, T_out, Q_internal, Q_solar, Q_HVAC):
        """
        Advance the model by one time step using Euler integration.

        Returns
        -------
        T_in_next : float
            Indoor temperature at the next time step [°C]
        energy : float
            Energy consumed by HVAC in this step [J]
            (positive = heating, negative = cooling)
        """
        # Heat balance
        dT = (Q_internal + Q_solar + Q_HVAC - self.UA * (T_in - T_out)) / self.C

        T_in_next = T_in + dT * self.dt

        # Energy used in this step (Joules)
        energy = Q_HVAC * self.dt

        return T_in_next, energy


# -------------------------------------------------
# Quick test (you can run this file to check it works)
# -------------------------------------------------
if __name__ == "__main__":
    model = BuildingThermalModel()

    T_in = 21.0          # starting indoor temperature
    T_out = 5.0          # cold outdoor temperature
    Q_internal = 500     # people + equipment [W]
    Q_solar = 0          # night time
    Q_HVAC = 2000        # heating power [W]

    T_next, energy = model.step(T_in, T_out, Q_internal, Q_solar, Q_HVAC)

    print(f"Next indoor temperature: {T_next:.2f} °C")
    print(f"Energy used this step:   {energy/1e6:.3f} MJ")