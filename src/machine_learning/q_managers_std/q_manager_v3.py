from src.parameters import *
import numpy as np
from src.math_tools import decision, numpy_object_array
import os

"""Multi-agent rollout Q-Learning (Bertsekas) + Double Q-Learning"""

class QManager:
    last_actions_idxs = None

    def __init__(self, uavs, uav_locs, energy_levels, reliability_levels, served_users_getter,
                 testing_flag=QLearningParams.TESTING_FLAG,
                 save_on=QLearningParams.SAVE_ON, load_model=QLearningParams.LOAD_MODEL):
        self.uav_locs = [numpy_object_array(uav_locs[i], Coords3d) for i in range(len(uav_locs))]
        self.energy_levels = np.array(energy_levels)
        self.reliability_levels = np.array(reliability_levels)
        self.reliability_quantizer = QLearningParams.RELIABILITY_LEVELS_QUANTIZER
        self.qs = np.zeros((len(uav_locs), len(reliability_levels), len(uav_locs)))
        self.uavs = uavs
        self.n_uavs = len(uavs)
        self.location_states = self.get_location_states()
        self.energy_states = self.get_energy_states()
        self.get_n_served_users = served_users_getter
        self.reliability_states, self.n_unserved = self.get_reliability_levels()
        self.testing_flag = testing_flag
        self.save_on = save_on  # Every N cycles


        if not load_model:
            self.qs = [[], []]
            qs_shape = [self.uav_locs[0].size] * self.n_uavs + [
                self.reliability_levels.size] * self.n_uavs + [self.uav_locs[0].size]  # discarding energy for now
            for uav_idx in range(self.n_uavs):
                self.qs[0].append(np.zeros(tuple(qs_shape)))
                self.qs[1].append(np.zeros(tuple(qs_shape)))
                qs_shape += [self.uav_locs[0].size]
        else:
            self.qs = [[], []]
            file_name = os.path.join(QLearningParams.CHECKPOINTS_FILE, self.__repr__() + '.npz')
            npzfile = np.load(file_name)
            for idx, _arr in enumerate(npzfile.files):
                if idx < len(npzfile.files) / 2:
                    self.qs[0].append(npzfile[_arr])
                else:
                    self.qs[1].append(npzfile[_arr])

    def begin_cycle(self, cycle_idx):
        new_locations, self.last_actions_idxs = self.select_actions(cycle_idx)
        return new_locations

    def end_cycle(self, cycle_idx, simulation_steps_count):
        self.simulation_steps_count = simulation_steps_count
        self.update_states()
        reward = self.get_reward()
        self.update_q_values(reward, cycle_idx)
        if self.save_on and not (cycle_idx % self.save_on):
            self.save_checkpoint()

    def save_checkpoint(self, folder_name=QLearningParams.CHECKPOINTS_FILE):
        file_name = os.path.join(folder_name, self.__repr__())
        np.savez(file_name, *self.qs[0], *self.qs[1])

    def update_q_values(self, reward, cycle_idx):
        q_set_idx = np.random.randint(2)
        _, new_actions_idxs = self.select_actions(cycle_idx, testing=True, q_set=q_set_idx)
        learning_rate = QManager.get_learning_rate(cycle_idx)
        discount_ratio = QManager.get_discount_ratio()
        for idx in range(self.n_uavs):
            last_action_idx = tuple(self.last_actions_idxs[idx])
            new_action_idx = tuple(new_actions_idxs[idx])
            new_action_q = self.qs[1 - q_set_idx][idx][new_action_idx]
            old_q_value = self.qs[q_set_idx][idx][last_action_idx]
            updated_q_value = old_q_value + learning_rate * (reward + discount_ratio * new_action_q - old_q_value)
            self.qs[q_set_idx][idx][last_action_idx] = updated_q_value

    def update_states(self):
        self.location_states = [_list[-1] for _list in self.last_actions_idxs]
        self.energy_states = self.get_energy_states()
        self.reliability_states, self.n_unserved = self.get_reliability_levels()

    def select_actions(self, cycle_idx, testing=False, q_set=None):
        locations_idxs = [self.location_states[i] for i in range(self.n_uavs)]
        reliability_idxs = [self.reliability_states[i] for i in range(self.n_uavs)]
        actions_idxs = []
        new_locations = []
        actions_last_idxs = []
        for idx in range(self.n_uavs):
            q_idx = locations_idxs + reliability_idxs + actions_last_idxs
            q_values = self.qs[0][idx][tuple(q_idx)] + self.qs[1][idx][tuple(q_idx)] if q_set is None else\
                self.qs[q_set][idx][tuple(q_idx)]
            action_idx = QManager.select_action_from_qs(q_values, cycle_idx, testing=testing or self.testing_flag)
            actions_last_idxs.append(action_idx)
            actions_idxs.append(list(q_idx) + [action_idx])
            new_locations.append(self.uav_locs[idx][action_idx])
        return new_locations, actions_idxs

    @staticmethod
    def select_action_from_qs(q_values, cycle_idx, testing=False):
        if not testing and decision(QManager.get_exploration_probability(cycle_idx)):
            idx = np.random.randint(q_values.shape[0])
        else:
            idx = np.random.choice(np.where(q_values == q_values.max())[0])
        return idx

    @staticmethod
    def get_learning_rate(cycle_idx):
        return QLearningParams.LEARNING_RATE(cycle_idx)

    @staticmethod
    def get_exploration_probability(cycle_idx):
        return QLearningParams.EXPLORATION_PROB(cycle_idx)

    @staticmethod
    def get_discount_ratio():
        return QLearningParams.DISCOUNT_RATIO

    def get_location_states(self):
        states = np.zeros((self.n_uavs,), dtype=int)
        for idx, _uav in enumerate(self.uavs):
            states[idx] = np.where([self.uav_locs[idx][i] == self.uavs[idx].coords for i in range(self.uav_locs[0].shape[0])])[0][0]
        return states

    def get_energy_states(self):
        states = np.zeros((self.n_uavs,), dtype=int)
        for idx, _uav in enumerate(self.uavs):
            states[idx] = np.sum(_uav.battery.energy_level > self.energy_levels)
        # print("Energies: ", states)
        return states

    def get_reliability_levels(self):
        users_stats = self.get_n_served_users()
        over_capacity = (users_stats[0][0] - MAX_USERS_PER_DRONE > 0) * (users_stats[0][0] - MAX_USERS_PER_DRONE)
        unsatisfied_sinr = users_stats[0][1]
        # print("User stats: ", users_stats, "\n Rewards: ", np.ceil(np.log2(np.maximum(over_capacity.T, unsatisfied_sinr.T))).astype(int))
        max_values = np.maximum(over_capacity.T, unsatisfied_sinr.T)
        return np.ceil(np.log(max_values, where=max_values > 0)/self.reliability_quantizer).astype(int), max_values

    def get_reward(self):
        return -np.sum(self.n_unserved)

    def get_n_served_users(self):
        """Defined by caller"""
        pass

    def reset_states(self):
        self.location_states = self.get_location_states()
        self.energy_states = self.get_energy_states()
        self.reliability_states, self.n_unserved = self.get_reliability_levels()

    @staticmethod
    def __repr__():
        return f'QManager_V3_{QLearningParams.CHECKPOINT_ID}'
