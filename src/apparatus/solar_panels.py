from pysolar.solar import *
from dateutil import tz
import datetime
import pysolar
from src.data_structures import Coords3d
from src.parameters import *
from shapely.geometry import polygon, point
from shapely.affinity import translate
import numpy as np
from src.math_tools import rotate, line_point_angle, point_inside_prlgm
from time import time


class SolarPanel:
    def __init__(self, coords: Coords3d, panel_manager):
        self.coords = coords
        self.cloud = None
        self.efficiency_factor = PANEL_EFFICIENCY_FACTOR
        self.panel_area = SOLAR_PANEL_AREA
        self.panel_manager = panel_manager

    def get_cloud_attenuation(self):
        if self.cloud is None:
            return 1
        else:
            distance_cloud = min(max(self.cloud.z_range[1] - self.coords.z, 0),
                                 self.cloud.z_range[1] - self.cloud.z_range[0])
        return np.exp(-distance_cloud*ABSORPTION_COEFFICIENT_CLOUD)

    def get_generated_power(self):
        return self.panel_manager.current_irradiation * self.panel_area * self.efficiency_factor *\
               self.get_cloud_attenuation()


class IrradiationManager:
    def __init__(self, latitude=40.418725331325, longitude=-3.704271435627907, date_time=None):
        self.latitude = latitude
        self.longitude = longitude
        if date_time is not None:
            self.date_time = date_time
        else:
            self.date_time = datetime.datetime(2022, STARTING_MONTH, STARTING_DAY, STARTING_HOUR,
                                               STARTING_MINUTE,
                                               0, tzinfo=tz.gettz(TIME_ZONE))

    def recalculate_sun_irradiation(self, date_time=None):
        if date_time is None:
            self.sun_altitude = get_altitude(self.latitude, self.longitude, self.date_time)
            self.current_irradiation = radiation.get_radiation_direct(self.date_time, self.sun_altitude)
        else:
            self.sun_altitude = get_altitude(self.latitude, self.longitude, date_time)
            self.current_irradiation = radiation.get_radiation_direct(date_time, self.sun_altitude)


class Cloud:
    def __init__(self, _polygon: polygon.Polygon, z_range=None):
        self.polygon_2d = _polygon
        self.z_range = z_range

    def check_overlap(self, other_coords: Coords3d):
        _point = point.Point(other_coords.x, other_coords.y)
        return self.polygon_2d.intersects(_point)

    def move(self, speed=CLOUD_SPEED, direction=(0, 1), t_duration=1):
        theta = np.arctan(direction[0] / direction[1])
        x_offset = speed * t_duration * np.sin(theta)
        y_offset = speed * t_duration * np.cos(theta)
        self.polygon_2d = translate(self.polygon_2d, x_offset, y_offset)


if __name__ == "__main__":
    src_x, src_y = 15, 20

    _point = point.Point(15, 20)
    a = polygon.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    a.intersects(_point)

    p1, p1p, p2p, p2 = (0, 0), (20, 0), (20, 20), (0, 20)
    point_inside_prlgm(src_x, src_y, [p1, p1p, p2p, p2])

    time1 = time()

    for i in range(5000000):
        a.intersects(_point)

    time2 = time()
    time_s = time2 - time1
    print(time_s)

    for i in range(5000000):
        point_inside_prlgm(src_x, src_y, [p1, p1p, p2p, p2])
    time_ns = time() - time2
    print(time_ns)
