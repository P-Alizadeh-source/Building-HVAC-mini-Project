"""
Reduced-order 2R-2C building thermal model.

The model represents:
- indoor air temperature
- building thermal mass temperature
- heat transfer to outdoor air
- internal gains
- solar gains
- HVAC thermal power

Equations
---------
C_air dT_air/dt =
    (T_out - T_air) / R_out
    + (T_mass - T_air) / R_mass
    + Q_internal
    + Q_solar
    + Q_HVAC

C_mass dT_mass/dt =
    (T_air - T_mass) / R_mass
"""

import numpy as np


class BuildingThermalModel:
    def __init__(
        self, 
        C_air=5.0e6,
        C_mass=5.0e7,
        R_out=0.02,
        R_mass=0.05,
        dt=3600,
    ):

        """
         Parameters
        ----------
        C_air : float
            Thermal capacity of indoor air [J/K]

        C_mass : float
            Effective thermal capacity of building mass [J/K]

        R_out : float
            Thermal resistance between indoor air and outdoors [K/W]

        R_mass : float
            Thermal resistance between indoor air and thermal mass [K/W]

        dt : float
            Simulation time step [s]
        """

        self.C_air = C_air
        self.C_mass = C_mass
        self.R_out = R_out
        self.R_mass = R_mass
        self.dt = dt

    def step(self,
        T_air,
        T_mass,
        T_out,
        Q_internal,
        Q_solar,
        Q_HVAC):

        """
        Advance the building model by one time step.

        Parameters
        ----------
        T_air : float
            Indoor air temperature [°C]

        T_mass : float
            Building thermal mass temperature [°C]

        T_out : float
            Outdoor temperature [°C]

        Q_internal : float
            Internal heat gains [W]

        Q_solar : float
            Solar heat gains [W]

        Q_HVAC : float
            HVAC thermal power [W]
            Positive = heating
            Negative = cooling

        Returns
        -------
        T_air_next : float
            Next indoor air temperature [°C]

        T_mass_next : float
            Next building mass temperature [°C]
        """

        # Heat transfer from outdoors to indoor air
        Q_out = (T_out - T_air) / self.R_out

        # Heat transfer between thermal mass and indoor air
        Q_mass = (T_mass - T_air) / self.R_mass

        # Indoor air energy balance
        dT_air_dt = (
            Q_out
            + Q_mass
            + Q_internal
            + Q_solar
            + Q_HVAC
        ) / self.C_air

        # Thermal mass energy balance
        dT_mass_dt = (
            (T_air - T_mass) / self.R_mass
        ) / self.C_mass

        # Euler integration
        T_air_next = T_air + dT_air_dt * self.dt
        T_mass_next = T_mass + dT_mass_dt * self.dt

        return T_air_next, T_mass_next


# -------------------------------------------------
# Quick test (you can run this file to check it works)
# -------------------------------------------------
if __name__ == "__main__":

    model = BuildingThermalModel()

    T_air = 21.0
    T_mass = 21.0
    T_out = 5.0

    Q_internal = 500.0
    Q_solar = 0.0
    Q_HVAC = 2000.0

    T_air_next, T_mass_next = model.step(
        T_air=T_air,
        T_mass=T_mass,
        T_out=T_out,
        Q_internal=Q_internal,
        Q_solar=Q_solar,
        Q_HVAC=Q_HVAC,
    )

    print(
        f"Next indoor air temperature: "
        f"{T_air_next:.2f} °C"
    )

    print(
        f"Next thermal mass temperature: "
        f"{T_mass_next:.2f} °C"
    )