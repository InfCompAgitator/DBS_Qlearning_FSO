from src.environment.user_mobility import ObstaclesMobilityModel
from src.users import User
from src.drone_station import DroneStation
from src.macro_base_station import MacroBaseStation
import numpy as np
from src.parameters import TX_POWER_FSO_MBS, TX_POWER_FSO_DRONE, NUM_UAVS, NUM_MBS, NUM_OF_USERS, \
    USER_MOBILITY_SAVE_NAME, TIME_STEP, USER_SINR_COVERAGE_HIST, DRONE_HEIGHT, SKIP_ENERGY_UPDATE, QLearningParams,\
    USER_SPEED_DIVISOR
from src.data_structures import Coords3d
from src.apparatus.solar_panels import IrradiationManager
from src.apparatus.fso_transceiver import FsoTransceiver
from src.types_constants import LinkType
from multiprocessing import Manager
from time import sleep
from src.math_tools import lin2db


class SimulationController:
    plot_flag = False
    bs_rf_list = []
    mobility_model = None
    steps_count = 1
    users_per_bs = None

    def __init__(self, initial_uavs_coords=None, solar_panels=False, n_users=NUM_OF_USERS, speed_divisor=USER_SPEED_DIVISOR):
        self.speed_divisor = speed_divisor
        self.reset_mobility_model(n_users=n_users)
        self.users = [User(self.mobility_model.users[i])
                      for i in range(self.mobility_model.n_users)]
        self.base_stations = [MacroBaseStation(mbs_id=0)]  # At 0,0,MBS_HEIGHT
        self.area_bounds = self.mobility_model.area_bounds
        self.irradiation_manager = IrradiationManager() if solar_panels else None
        self.create_drone_stations(check_obstacles=False, irradiation_manager=self.irradiation_manager,
                                   initial_coords_input=initial_uavs_coords)
        self.link_stations_with_fso_star()
        self.init_users_rfs()
        self.stations_list = []

    def reset_mobility_model(self, load=True, n_users=NUM_OF_USERS):
        if self.mobility_model is None:
            self.mobility_model = ObstaclesMobilityModel(number_of_users=n_users)
            # Comment below to not load
            if load:
                self.mobility_model.load_model(save_name=USER_MOBILITY_SAVE_NAME, max_num_users=n_users,
                                               speed_divisor=self.speed_divisor)
            else:
                self.mobility_model.reset_users()
        else:
            self.mobility_model.reset_users(self.speed_divisor)
            self.init_users_rfs()

    def simulate_time_step(self, time_step=None, static_users=False):
        self.steps_count += 1
        users_moving = self.mobility_model.generate_model_step(time_step, static_users)
        self.move_uavs()
        self.update_fso_stats()
        if self.irradiation_manager:
            self.irradiation_manager.recalculate_sun_irradiation(self.mobility_model.time_of_day)
        self.update_drones_energy()
        self.update_users_rfs()
        return users_moving

    def update_sinr_coverage_scores(self):
        [_user.rf_transceiver.update_sinr_coverage_score(self.steps_count) for _user in self.users]

    def update_fso_stats(self):
        if SKIP_ENERGY_UPDATE:
            return
        [_drone.update_fso_status() for _drone in self.base_stations[NUM_MBS:]]

    def move_uavs(self):
        [_drone.move(TIME_STEP) for _drone in self.base_stations[NUM_MBS:]]
        if self.plot_flag:
            self.update_bs_shared_coords()

    def get_users_stats(self):
        return np.array([_user.rf_transceiver.get_stats() for _user in self.users])

    def get_uavs_energy(self):
        return [_drone.battery.energy_level for _drone in self.base_stations[NUM_MBS:]]

    def update_drones_energy(self):
        if SKIP_ENERGY_UPDATE:
            return
        [_drone.update_energy(self.mobility_model.time_step) for _drone in self.base_stations[NUM_MBS:]]

    def get_users_rf_means(self):
        return np.array([[lin2db(_user.rf_transceiver.mean_sinr) for _user in self.users],
                         [_user.rf_transceiver.mean_capacity for _user in self.users]])

    def get_uavs_total_consumed_energy(self):
        return sum([_drone.battery.get_total_energy_consumption() for _drone in self.base_stations[NUM_MBS:]])

    def set_ues_base_stations(self, exclude_mbs=True):
        self.bs_rf_list = [_bs.rf_transceiver for _bs in self.base_stations[NUM_MBS if exclude_mbs else 0:]]
        all_freqs = [_bs.carrier_frequency for _bs in self.bs_rf_list]
        available_freqs = set(all_freqs)
        self.stations_list = []
        for _freq in available_freqs:
            bs_list = []
            for idx, _freq_bs in enumerate(all_freqs):
                if _freq_bs == _freq:
                    bs_list.append(self.bs_rf_list[idx])
            self.stations_list.append(bs_list)
        for _user in self.users:
            _user.rf_transceiver.set_available_base_stations(self.stations_list)

    def link_stations_with_fso_star(self):
        for i in range(NUM_MBS, len(self.base_stations)):
            self.link_two_stations_with_fso(self.base_stations[0], self.base_stations[i])

    def link_stations_with_fso_sequential(self):
        mbs_tr = FsoTransceiver(coords=self.base_stations[0].coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                bs_id=self.base_stations[0].id)
        dbs_tr = FsoTransceiver(coords=self.base_stations[1].coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                bs_id=self.base_stations[1].id, is_backhaul=True, endpoint=mbs_tr)
        mbs_tr.endpoint = dbs_tr
        self.base_stations[0].fso_transceivers.append(mbs_tr)
        self.base_stations[1].fso_transceivers.append(dbs_tr)

        for i in range(1, len(self.base_stations) - 1):
            self.link_two_stations_with_fso(self.base_stations[i], self.base_stations[i + 1])
        self.update_fso_stats()

    @staticmethod
    def link_two_stations_with_fso(backhaul_bs, next_bs):
        """First is backhaul second is normal"""
        if isinstance(backhaul_bs, MacroBaseStation) and isinstance(next_bs, MacroBaseStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.G2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.G2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, MacroBaseStation) and isinstance(next_bs, DroneStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, DroneStation) and isinstance(next_bs, DroneStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2A, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2A, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)
        elif isinstance(backhaul_bs, DroneStation) and isinstance(next_bs, MacroBaseStation):
            bdbs_tr = FsoTransceiver(coords=backhaul_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_DRONE,
                                     bs_id=backhaul_bs.id, is_backhaul=False)
            ndbs_tr = FsoTransceiver(coords=next_bs.coords, link_type=LinkType.A2G, t_power=TX_POWER_FSO_MBS,
                                     bs_id=next_bs.id, is_backhaul=True, endpoint=bdbs_tr)

        bdbs_tr.endpoint = ndbs_tr
        backhaul_bs.fso_transceivers.append(bdbs_tr)
        next_bs.fso_transceivers.append(ndbs_tr)

    def create_drone_stations(self, check_obstacles=False, irradiation_manager=None, initial_coords_input=None):
        for i in range(NUM_UAVS):
            if initial_coords_input is None:
                initial_coords = np.random.uniform(low=self.area_bounds.T[0], high=self.area_bounds.T[1], size=2)
                drone_height = DRONE_HEIGHT
            else:
                initial_coords = initial_coords_input[i]
                drone_height = initial_coords_input[i].z

            initial_coords = Coords3d(initial_coords[0], initial_coords[1], drone_height)
            # initial_coords = Coords3d(2, 215, 25)
            new_station = DroneStation(coords=initial_coords, irradiation_manager=irradiation_manager,
                                       drone_id=i + 1)  # ,
            # carrier_frequency=UAVS_FREQS[i])
            if check_obstacles:
                while self.mobility_model.obstacles_objects.check_overlap(new_station.coords):
                    initial_coords = np.random.uniform(low=self.area_bounds.T[0], high=self.area_bounds.T[1], size=2)
                    initial_coords = Coords3d(initial_coords[0], initial_coords[1], DRONE_HEIGHT)
                    new_station.coords.set(initial_coords)
            self.base_stations.append(new_station)
        self.base_stations = tuple(self.base_stations)

    def set_drones_waypoints(self, waypoints, speed):
        for i in range(NUM_UAVS):
            duration_to_reach = self.base_stations[NUM_MBS + i].coords.get_distance_to(waypoints[i]) / speed
            if duration_to_reach > 0:
                self.base_stations[NUM_MBS + i].set_waypoint(waypoints[i], duration_to_reach)
        # # Random
        # for i in range(NUM_UAVS):
        #     new_loc = np.random.uniform(low=self.area_bounds.T[0], high=self.area_bounds.T[1], size=2)
        #     new_loc = Coords3d(new_loc[0], new_loc[1], np.random.uniform(low=0, high=200))
        #     self.base_stations[NUM_MBS + i].set_waypoint(new_loc, duration_to_reach)

    def add_base_stations_to_plot(self):
        self.base_stations_shared_coords = Manager().list()
        for _bs in self.base_stations:
            self.base_stations_shared_coords.append([_bs.coords.x, _bs.coords.y])
        self.mobility_model.add_base_stations_to_plot(self.base_stations_shared_coords)

    def update_bs_shared_coords(self):
        for idx, _bs in enumerate(self.base_stations):
            self.base_stations_shared_coords[idx] = [_bs.coords.x, _bs.coords.y]

    def generate_plot(self):
        self.add_base_stations_to_plot()
        self.mobility_model.generate_plot()
        self.plot_flag = True

    def update_users_per_bs(self):
        if ((self.steps_count - 1) * TIME_STEP) % 1 == 0:
            bs_list = [[] for i in range(len(self.bs_rf_list))]
            for _user in self.users:
                bs_id, sinr, snr, rx_power = _user.rf_transceiver.get_serving_bs_info(recalculate=False)
                bs_list[bs_id - NUM_MBS].append(_user)
            for _bs, _users in zip(self.bs_rf_list, bs_list):
                _bs.n_associated_users = len(_users)
            self.users_per_bs = bs_list

    def get_users_per_bs(self):
        return self.users_per_bs

    def get_uavs_locs(self):
        coords = []
        for idx, _bs in enumerate(self.base_stations[NUM_MBS:]):
            coords.append(_bs.coords)
        return coords

    def init_users_rfs(self):
        [_user.rf_transceiver.sinr_coverage_history.fill(0) for _user in self.users]
        for _user in self.users:
            _user.rf_transceiver.sinr_coverage_score = 0
        self.set_ues_base_stations()
        [_user.rf_transceiver.get_serving_bs_info(recalculate=True) for _user in self.users]
        self.update_sinr_coverage_scores()
        self.update_users_per_bs()
        self.init_users_rf_stats()

    def init_users_rf_stats(self):
        [_user.rf_transceiver.init_rf_stats() for _user in self.users]

    def update_users_rfs(self):
        [_user.rf_transceiver.get_serving_bs_info(recalculate=True) for _user in self.users]
        self.update_sinr_coverage_scores()
        self.update_users_per_bs()
        if QLearningParams.TESTING_FLAG:
            [_user.rf_transceiver.update_rf_stats(self.steps_count) for _user in self.users]


if __name__ == "__main__":
    a = SimulationController()
    # a.generate_plot()
    # a.set_drones_waypoints(TIME_STEP*20)
    while 1:
        a.simulate_time_step()
        sleep(1)
        print("======================================================")
        for _uav in a.base_stations[NUM_MBS:]:
            # print(_uav.battery.energy_level)
            # print(_uav.battery.recharge_count)
            # print(_uav.fso_transceivers[0].received_charge_power)
            # print(_uav.fso_transceivers[0].link_capacity)
            _uav.coords.update(Coords3d(500, 500, 200), 5)
            # print(_uav.coords)
            print("distance:", _uav.coords.get_distance_to(a.base_stations[0].coords))
        # for _user in a.users:
        # _user.rf_transceiver.get_serving_bs_info(recalculate=True)
        # bs_id, sinr, snr, rx_power, capacity = _user.rf_transceiver.get_serving_bs_info(recalculate=True)
        # print(lin2db(sinr))
        # print(_user.coords.get_distance_to(_user.rf_transceiver.serving_bs.coords))
        # print(_user.coords)

    # a.mobility_model.generate_model()
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # a.simulate_time_step()
    # for _user in a.users:
    #     # _user.rf_transceiver.get_serving_bs_info(recalculate=True)
    #     # print(lin2db(_user.rf_transceiver.get_serving_bs_info(recalculate=True)[1]))
    #     # print(_user.coords.get_distance_to(_user.rf_transceiver.serving_bs.coords))
    #     # print(_user.rf_transceiver.serving_bs.coords)
    #     print(_user.coords)
    #     print("======================================================")
