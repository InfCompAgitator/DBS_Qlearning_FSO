from src.parameters import *
from src.main_controller import SimulationController
import numpy as np
import itertools
from tqdm import tqdm
FAIRNESS = True
if __name__ == "__main__":
    UAVS_HEIGHTS = [60, 80]#, 100]
    possible_uavs_locs = [[Coords3d(loc[i][0], loc[i][1], UAVS_HEIGHTS[j]) for i in range(len(loc))
                           for j in range(len(UAVS_HEIGHTS))] for loc in UAVS_LOCATIONS]

    initial_coords = [Coords3d(loc[0][0], loc[0][1], UAVS_HEIGHTS[0]) for loc in UAVS_LOCATIONS]
    simulation_controller = SimulationController(initial_coords)

    def get_score(fairness=FAIRNESS):
        users_per_bs = simulation_controller.get_users_per_bs()
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
        users_stats = [np.array(n_associated_users), np.array(n_unserved)], 1 - np.sum(np.array(n_unserved)) / NUM_OF_USERS,\
               coverage_scores
        if not fairness:
            over_capacity = (users_stats[0][0] - MAX_USERS_PER_DRONE > 0) * (users_stats[0][0] - MAX_USERS_PER_DRONE)
            unsatisfied_sinr = users_stats[0][1]
            max_values = np.maximum(over_capacity.T, unsatisfied_sinr.T)
            return np.ceil(max_values, where=max_values > 0).astype(int), max_values
        else:
            n_users = users_stats[0][0].sum()
            np.sum([_array.sum() for _array in users_stats[2]]) ** 2
            np.sum([(_array ** 2).sum() for _array in users_stats[2]])

            per_uav_sum = np.array([_array.sum() for _array in users_stats[2]])
            per_uav_squared_sum = np.array([(_array ** 2).sum() for _array in users_stats[2]])
            per_uav_score = per_uav_sum ** 2 / (users_stats[0][0] * per_uav_squared_sum)
            total_score = per_uav_sum.sum() ** 2 / (n_users * per_uav_squared_sum.sum())
            fairness_levels = np.floor(QLearningParams.FAIRNESS_EXP ** per_uav_score)
            return fairness_levels, np.array(total_score) * users_stats[1]

    min_scores = []
    best_locs =[]
    best_locs_idxs =[]
    static_scores =[]
    for i in tqdm(range(100)):
        for j in range(2):
            simulation_controller.simulate_time_step(TIME_STEP)
        min_score = get_score()[1].sum()
        static_scores.append(min_score)
        if not FAIRNESS:
            for locs_idx, locs in enumerate(list(itertools.product(possible_uavs_locs[0], possible_uavs_locs[1], possible_uavs_locs[2],
                                                                   possible_uavs_locs[3]))):#, possible_uavs_locs[4], possible_uavs_locs[5],
                                                                   #possible_uavs_locs[6])))):
                for idx in range(NUM_UAVS):
                    simulation_controller.base_stations[NUM_MBS+idx].coords.set(locs[idx])
                score = get_score()
                if score[1].sum() < min_score or not locs_idx:
                    min_score = score[1].sum()
                    n_users = score[1]
                    locs_win = locs
                    locs_idx_win = locs_idx
            min_scores.append(min_score)
            best_locs.append(locs_win)
            best_locs_idxs.append(locs_idx_win)
    print(min_score)
