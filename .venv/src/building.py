"""Simple 2R-2C building model."""
import numpy as np

class Building:
    def __init__(self, dt=3600, C_air=5e6, C_mass=5e7, R_out=0.02, R_mass=0.05):
        self.dt, self.C_air, self.C_mass = dt, C_air, C_mass
        self.R_out, self.R_mass = R_out, R_mass

    def step(self, T_air, T_mass, T_out, Q_internal, Q_solar, Q_hvac):
        Q_out = (T_out - T_air) / self.R_out
        Q_mass = (T_mass - T_air) / self.R_mass
        dT_air = (Q_out + Q_mass + Q_internal + Q_solar + Q_hvac) / self.C_air
        dT_mass = ((T_air - T_mass) / self.R_mass) / self.C_mass
        return T_air + dT_air*self.dt, T_mass + dT_mass*self.dt

if __name__ == '__main__':
    b=Building(); print(b.step(21,21,5,500,0,2000))
