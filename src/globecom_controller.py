import numpy as np

from src.main_controller import SimulationController
from src.parameters import *
from src.machine_learning.q_managers_std.q_manager import QManager as QM0
from src.machine_learning.q_managers_std.q_manager_v0 import QManager as QM1
from src.machine_learning.q_managers_std.q_manager_v2 import QManager as QM2
from src.machine_learning.q_managers_std.q_manager_v3 import QManager as QM3

from src.machine_learning.q_managers_1_history.q_manager import QManager as QMH0
from src.machine_learning.q_managers_1_history.q_manager_v0 import QManager as QMH1
from src.machine_learning.q_managers_1_history.q_manager_v2 import QManager as QMH2
from src.machine_learning.q_managers_1_history.q_manager_v3 import QManager as QMH3

from src.machine_learning.q_managers_fairness.q_manager_v3_std import QManager as QMF3
from src.machine_learning.q_managers_fairness.q_manager_v3_h import QManager as QMFH3
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time

N_ITERATIONS = 500
N_CYCLES = int(SIMULATION_TIME / QLearningParams.TIME_STEP_Q - 1)
HISTORY_DIR = 'C:\\Users\\user\\PycharmProjects\\droneFsoCharging\\src\\machine_learning\\run_history'
HISTORY_ID = '0' + QLearningParams.CHECKPOINT_ID

STATS_UPDATE_INTERVAL = 5

def q_managers_dict(x):
    return {
        '00': QM0, '01': QM1, '02': QM2, '03': QM3,
        'H0': QMH0, 'H1': QMH1, 'H2': QMH2, 'H3': QMH3,
        'Q Learning': QMF3, 'FH3': QMFH3
    }.get(x, QM0)


scores_levels = list(range(int(np.ceil(np.log(NUM_OF_USERS) / QLearningParams.RELIABILITY_LEVELS_QUANTIZER))))
scores_levels = list(range(QLearningParams.FAIRNESS_LEVELS + 1))


class GlobecomController:

    def __init__(self, initial_coords_idxs=(0, 0), q_manager_id='Q Learning', n_users=NUM_OF_USERS,
                 speed_divisor=USER_SPEED_DIVISOR, verbose=True):
        self.n_users = n_users
        initial_coords = [Coords3d(loc[initial_coords_idxs[0]][0], loc[initial_coords_idxs[0]][1],
                                   UAVS_HEIGHTS[initial_coords_idxs[0]]) for loc in UAVS_LOCATIONS]
        self.simulation_controller = SimulationController(initial_coords, n_users=n_users, speed_divisor=speed_divisor)
        self.configure_fso_transcievers()
        self.possible_uavs_locs = [[Coords3d(loc[i][0], loc[i][1], UAVS_HEIGHTS[j]) for i in range(len(loc))
                                    for j in range(len(UAVS_HEIGHTS))] for loc in UAVS_LOCATIONS]
        self.q_manager = q_managers_dict(q_manager_id)(self.simulation_controller.base_stations[NUM_MBS:],
                                                       self.possible_uavs_locs,
                                                       QLearningParams.ENERGY_LEVELS,
                                                       scores_levels,
                                                       self.get_n_served_users)
        self.verbose = verbose

        self.sinr_levels = np.arange(40, dtype=int)
        self.capacity_levels = np.arange(0, 20*1024, (20 * 1024)/100)

        self.sinr_stats = np.zeros((3, self.sinr_levels.shape[0] -1), dtype=int) #mean, max, min stats
        self.capacity_stats = np.zeros((3, self.capacity_levels.shape[0] -1), dtype=int)  # mean, max, min stats


    def configure_fso_transcievers(self):
        for idx, _uav in enumerate(self.simulation_controller.base_stations[NUM_MBS:]):
            _uav.fso_transceivers[0].endpoint.beamwaist_radius = BEAMWAIST_RADII[idx]
            _uav.fso_transceivers[0].tx_power = 0  # Only downlink is active

    def get_n_served_users(self):
        users_per_bs = self.simulation_controller.get_users_per_bs()
        n_associated_users = []
        n_unserved = []
        coverage_scores = []
        for uav_id, _users in enumerate(users_per_bs):
            cov_scores_uav = np.zeros(len(_users))
            n_unserved_uav = 0
            n_associated_users.append(len(_users))
            for user_idx, _user in enumerate(_users):
                n_unserved_uav += 1 - _user.rf_transceiver.is_sinr_satisfied()
                cov_scores_uav[user_idx] = _user.rf_transceiver.sinr_coverage_score
            coverage_scores.append(cov_scores_uav)
            n_unserved.append(n_unserved_uav)
        return [np.array(n_associated_users), np.array(n_unserved)], 1 - np.sum(np.array(n_unserved)) / self.n_users, \
               coverage_scores

    def perform_cycle(self, cycle_idx, testing=QLearningParams.TESTING_FLAG):
        cycle_stats = None
        new_uav_locs = self.q_manager.begin_cycle(cycle_idx)
        self.simulation_controller.set_drones_waypoints(new_uav_locs, UAV_TRAVEL_SPEED)
        for i in range(int(QLearningParams.TIME_STEP_Q / TIME_STEP)):
            self.simulation_controller.simulate_time_step(TIME_STEP)
            if QLearningParams.TESTING_FLAG and (i*TIME_STEP) % STATS_UPDATE_INTERVAL == 0:
                stats = self.simulation_controller.get_users_stats().transpose()
                self.sinr_stats[0] += np.histogram(stats[3], self.sinr_levels)[0]
                self.sinr_stats[1] += np.histogram(stats[4], self.sinr_levels)[0]
                self.sinr_stats[2] += np.histogram(stats[5], self.sinr_levels)[0]
                self.capacity_stats[0] += np.histogram(stats[0], self.capacity_levels)[0]
                self.capacity_stats[1] += np.histogram(stats[1], self.capacity_levels)[0]
                self.capacity_stats[2] += np.histogram(stats[2], self.capacity_levels * 1024)[0]
                self.simulation_controller.init_users_rf_stats()


        # for _uav in self.simulation_controller.base_stations[NUM_MBS:]:
        #     assert (_uav.moving_flag == False)
        self.q_manager.end_cycle(cycle_idx, self.simulation_controller.steps_count)

    def run_n_cycles(self, n_cycles=N_CYCLES):
        rewards = np.zeros((n_cycles,))
        fairnesses = np.zeros((n_cycles,))
        reliabilities = np.zeros((n_cycles,))
        for cycle_idx in range(1, n_cycles + 1):
            # time1 = time()
            self.perform_cycle(cycle_idx)
            # time2 = time()
            # print(time2 - time1)
            rewards[cycle_idx - 1] = self.q_manager.get_reward()
            fairnesses[cycle_idx - 1], reliabilities[cycle_idx - 1] = self.get_fairness_and_reliability()
            if self.verbose and not cycle_idx % 20:
                print(self.q_manager.location_states)
                print("Cycle: ", cycle_idx, "Reward: ", rewards[cycle_idx - 1], " Avg of last 50: ",
                      rewards[max(0, cycle_idx - 50 - 1):cycle_idx].mean())
        return rewards, fairnesses, reliabilities

    def run_cycle_with_fixed_locations(self, alternating=True):
        if alternating:
            new_locs_idx = (np.array(self.q_manager.get_location_states()) + 1) % np.array(
                [len(self.possible_uavs_locs[i]) for i in range(len(self.possible_uavs_locs))])
            new_uav_locs = [self.possible_uavs_locs[i][j] for i, j in zip(range(new_locs_idx.size), new_locs_idx)]
            self.simulation_controller.set_drones_waypoints(new_uav_locs, UAV_TRAVEL_SPEED)
        for i in range(int(QLearningParams.TIME_STEP_Q / TIME_STEP)):
            self.simulation_controller.simulate_time_step(TIME_STEP)
            # for _uav in self.simulation_controller.base_stations[NUM_MBS:]:
            #     if i <4 and alternating:
            #         assert (_uav.moving_flag == True)
            if (i*TIME_STEP) % STATS_UPDATE_INTERVAL == 0:
                stats = self.simulation_controller.get_users_stats().transpose()
                self.sinr_stats[0] += np.histogram(stats[3], self.sinr_levels)[0]
                self.sinr_stats[1] += np.histogram(stats[4], self.sinr_levels)[0]
                self.sinr_stats[2] += np.histogram(stats[5], self.sinr_levels)[0]
                self.capacity_stats[0] += np.histogram(stats[0], self.capacity_levels)[0]
                self.capacity_stats[1] += np.histogram(stats[1], self.capacity_levels)[0]
                self.capacity_stats[2] += np.histogram(stats[2], self.capacity_levels * 1024)[0]
                self.simulation_controller.init_users_rf_stats()

        _, self.q_manager.n_unserved = self.q_manager.get_reliability_levels()
        fairness, reliability = self.get_fairness_and_reliability()
        return self.q_manager.get_reward(), fairness, reliability

    def reset_model(self, n_users):
        self.simulation_controller.reset_mobility_model(n_users=n_users)
        self.q_manager.reset_states()

    def get_fairness_and_reliability(self):
        users_stats = self.get_n_served_users()
        n_users = users_stats[0][0].sum()
        per_uav_sum = np.array([_array.sum() for _array in users_stats[2]])
        per_uav_squared_sum = np.array([(_array ** 2).sum() for _array in users_stats[2]])
        per_uav_score = per_uav_sum ** 2 / (users_stats[0][0] * per_uav_squared_sum) * (
                users_stats[0][0] - users_stats[0][1]) / users_stats[0][0]
        per_uav_score[np.isnan(per_uav_score)] = 0
        fairness = per_uav_sum.sum() ** 2 / (n_users * per_uav_squared_sum.sum())
        reliability = users_stats[1]
        return fairness, reliability


if __name__ == '__main__':

    _controller = GlobecomController()
    file_name = os.path.join(HISTORY_DIR, _controller.q_manager.__repr__() + f'_{HISTORY_ID}')
    # _controller.simulation_controller.generate_plot()
    rewards_total = np.zeros((N_CYCLES * N_ITERATIONS,))
    for _iter in tqdm(range(1, N_ITERATIONS + 1)):
        _controller.reset_model(n_users=NUM_OF_USERS)
        rewards = _controller.run_n_cycles(N_CYCLES)
        rewards_total[(_iter - 1) * N_CYCLES: (_iter) * N_CYCLES], _, _= rewards
        print("Iter:", _iter)
        np.save(file_name, rewards_total)

    rewards_total = np.load(file_name + '.npy')
    look_ahead_n = 100
    rewards_discounted_cumsum = np.zeros(N_CYCLES * N_ITERATIONS - look_ahead_n)
    for i in range(len(rewards_total) - look_ahead_n):
        sum = 0
        discount = 1
        for j in range(i, i + look_ahead_n):
            sum += rewards_total[j] * discount
            discount *= 1
        rewards_discounted_cumsum[i] = sum / look_ahead_n

    plt.plot(rewards_discounted_cumsum)
    ax = plt.gca()
    ax.set_xlabel('Cycle', fontsize=15)
    ax.set_ylabel('Average Reward', fontsize=15)
    plt.savefig(file_name + '.eps', format='eps')
    plt.show()

##################################################################
# rewards_total = np.zeros((N_CYCLES,))
# for _iter in range(1, N_ITERATIONS):
#     _controller.simulation_controller.reset_mobility_model()
#     rewards = _controller.run_n_cycles(N_CYCLES)
#     rewards_total = rewards_total + (rewards - rewards_total)/_iter
#     print("Iter:", _iter)

# file_name = os.path.join(HISTORY_DIR, QManager.__repr__() + f'_{HISTORY_ID}')
# np.save(file_name, rewards_total)
# rewards_total = np.load(file_name + '.npy')
#
# look_ahead_n = 100
# rewards_discounted_cumsum = np.zeros(N_CYCLES - look_ahead_n)
# for i in range(len(rewards_total) - look_ahead_n):
#     sum = 0
#     discount = 1
#     for j in range(i, i + look_ahead_n):
#         sum += rewards_total[j] * discount
#         discount *= 1
#     rewards_discounted_cumsum[i] = sum / look_ahead_n
#
# plt.plot(rewards_discounted_cumsum)
# ax = plt.gca()
# ax.set_xlabel(QManager.__repr__(), fontsize=15)
# ax.set_ylabel('Average Reward', fontsize=15)
# plt.show()

#
# _controller.perform_cycle()
# _controller.perform_cycle()
# _controller.perform_cycle()
# _controller.simulation_controller.get_uavs_energy()
# _controller.simulation_controller.simulate_time_step()
# users_per_bs = _controller.simulation_controller.get_users_per_bs()
# sinrs = []
# for _user in _controller.simulation_controller.users:
#     bs_id, sinr, snr, rx_power, capacity = _user.rf_transceiver.get_serving_bs_info(recalculate=True)
#     print(
#         f'Station ID: {bs_id}, SINR: {lin2db(sinr)}, SNR: {lin2db(snr)}, Throughput (Mbps): {capacity / (1024 * 1024)}')
#     sinrs.append(sinr)
#
#
# # _controller.simulation_controller.users[1].coords.get_distance_to(_controller.simulation_controller.users[1].rf_transceiver.stations_list[0][2].coords)
#
# # i = 0
# # for _user in users_per_bs[i]:
# #     bs_id, sinr, snr, rx_power, capacity = _user.rf_transceiver.get_serving_bs_info(recalculate=True)
# #     print(f'Station ID: {bs_id}, SINR: {lin2db(sinr)}, SNR: {lin2db(snr)}, Throughput (Mbps): {capacity / (1024*1024)}')
