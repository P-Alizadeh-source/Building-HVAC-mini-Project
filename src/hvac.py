"""
Simple HVAC electrical power model.

Q_HVAC is thermal heating/cooling power [W].
P_electric is electrical power consumption [W].
"""


class HVACSystem:

    def __init__(
        self,
        heating_cop=3.5,
        cooling_cop=3.0,
        fan_power=300.0,
    ):
        self.heating_cop = heating_cop
        self.cooling_cop = cooling_cop
        self.fan_power = fan_power

    def electrical_power(self, Q_HVAC):
        """
        Convert HVAC thermal power to electrical power.

        Positive Q_HVAC = heating
        Negative Q_HVAC = cooling
        """

        if Q_HVAC > 0:
            compressor_power = Q_HVAC / self.heating_cop

        elif Q_HVAC < 0:
            compressor_power = abs(Q_HVAC) / self.cooling_cop

        else:
            compressor_power = 0.0

        if Q_HVAC != 0:
            return compressor_power + self.fan_power

        return 0.0

    def electrical_energy_kWh(self, Q_HVAC, dt=3600):
        """
        Electrical energy consumed during one time step [kWh].
        """

        power = self.electrical_power(Q_HVAC)

        return power * dt / 3.6e6


if __name__ == "__main__":

    hvac = HVACSystem()

    print(
        "Heating:",
        hvac.electrical_power(5000),
        "W"
    )

    print(
        "Cooling:",
        hvac.electrical_power(-5000),
        "W"
    )