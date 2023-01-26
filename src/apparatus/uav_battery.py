from src.parameters import *
import numpy as np
from src.apparatus.solar_panels import SolarPanel
import time

"""Energy model based on https://ieeexplore.ieee.org/document/8648498. It is assumed that during each time slot
the UAV is moving with small accelaration and constant velocity"""


class UavBattery:
    rotor_disks_area = None
    solar_panel = None

    def __init__(self, starting_energy=UAV_STARTING_ENERGY, irradiation_manager=None, coords=None):
        """If irradiation_manager and coords are passed then a solar panel is constructed"""
        self.energy_level = starting_energy
        self.uav_mass = UAV_MASS
        self.set_rotor_disks_area()
        self.recharge_count = 0
        if irradiation_manager and coords:
            self.solar_panel = SolarPanel(coords, irradiation_manager)
        self.skip_movement_energy = False

    def update_energy(self, time_duration, speed_x=0, speed_y=0, speed_z=0, tx_power=0, fso_received_power=0):
        if SKIP_ENERGY_UPDATE:
            return
        # print("tx_power:", tx_power)
        self.discharge_energy(time_duration, speed_x, speed_y, speed_z, tx_power)
        # print("fso received power:", fso_received_power)
        self.recharge_energy(time_duration, fso_received_power)
        self.energy_level = min(self.energy_level, UAV_MAX_ENERGY)
        if self.energy_level <= UAV_MIN_ENERGY:
            self.energy_empty()
            return False
        return True

    def energy_empty(self):
        self.energy_level = UAV_STARTING_ENERGY + self.energy_level
        self.recharge_count += 1

    def get_total_energy_consumption(self):
        return self.recharge_count * UAV_STARTING_ENERGY + (UAV_STARTING_ENERGY - self.energy_level)

    def recharge_energy(self, time_duration, fso_received_power):
        if SKIP_ENERGY_CHARGE:
            return
        self.energy_level += fso_received_power * time_duration
        if self.solar_panel:
            self.energy_level += self.solar_panel.get_generated_power() * time_duration

    def discharge_energy(self, time_duration, speed_x=0, speed_y=0, speed_z=0, tx_power=0):
        self.energy_level -= self.get_dynamic_consumed_energy(time_duration, speed_x, speed_y, speed_z) + \
                             tx_power * time_duration

    def get_level_flight_power(self, speed_x=0, speed_y=0):
        uav_weight = self.uav_mass * GRAVITATION_ACCELERATION
        if abs(speed_x) + abs(speed_y) == 0:
            """Hovering Power"""
            return uav_weight ** (3 / 2) / np.sqrt(2 * AIR_DENSITY * self.rotor_disks_area)

        horizontal_speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
        _power = uav_weight ** 2 / (np.sqrt(2) * AIR_DENSITY * self.rotor_disks_area) / np.sqrt(
            horizontal_speed ** 2 + np.sqrt(
                horizontal_speed ** 4 + 4 * (np.sqrt(uav_weight / (2 * AIR_DENSITY * self.rotor_disks_area))) ** 4))
        return _power

    def get_vertical_flight_power(self, speed_z):
        return speed_z * self.uav_mass * GRAVITATION_ACCELERATION

    def get_blade_drag_power(self, speed_x=0, speed_y=0):
        return 1 / 8 * PROFILE_DRAG_COEFFICIENT * AIR_DENSITY * self.rotor_disks_area * np.sqrt(
            speed_x ** 2 + speed_y ** 2) ** 3

    def set_rotor_disks_area(self, area=None, propellers_radius=UAV_PROPELLER_RADIUS,
                             number_of_uav_propellers=NUMBER_OF_UAV_PROPELLERS):
        if area is None:
            self.rotor_disks_area = propellers_radius ** 2 * np.pi * number_of_uav_propellers
        else:
            self.rotor_disks_area = area

    def get_consumption_power(self, speed_x=0, speed_y=0, speed_z=0):
        _power = self.get_level_flight_power(speed_x, speed_y)
        if speed_x + speed_y != 0:
            _power += self.get_blade_drag_power(speed_x, speed_y)
        if speed_z == 0:
            return _power
        return _power + self.get_vertical_flight_power(speed_z)

    def get_dynamic_consumed_energy(self, time_duration, speed_x=0, speed_y=0, speed_z=0):
        if self.skip_movement_energy:
            self.skip_movement_energy = False
            return 0
        dynamic_power = self.get_consumption_power(speed_x, speed_y, speed_z)
        # print("dynamic consumed power:", dynamic_power)
        return time_duration * dynamic_power


if __name__ == '__main__':
    ub = UavBattery()
    time1 = time.time()
    for i in range(10000):
        ub.update_energy(5, 1, 2, 3, 4, 5)
    print(time.time() - time1)
