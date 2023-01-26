from shutil import rmtree
import os
from src.parameters import *
from src.globecom_controller import GlobecomController
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

results_id = 5
results_folder = f'C:\\Users\\user\\PycharmProjects\\droneFsoCharging\\src\\results\\{results_id}\\'

N_CYCLES = int(SIMULATION_TIME / QLearningParams.TIME_STEP_Q - 1 -50)
N_ITERS = 100

def moving_average(x, w):
    y_padded = np.pad(x, (w // 2, w - 1 - w // 2), mode='edge')
    return np.convolve(y_padded, np.ones((w,))/w, mode='valid')

if __name__ == '__main__':
    if os.path.isdir(results_folder):
        rmtree(results_folder)
    os.mkdir(results_folder)


    init_speed_divisor = 1
    speed_step = 4 / N_ITERS
    speed_divisors = np.linspace(0.25,5, N_ITERS)

    q_manager_id = 'Q Learning'
    num_users = [150, 200, 300]
    total_rewards_means = np.zeros((len(num_users), 3))
    total_fairness_means = np.zeros((len(num_users), 3))
    total_reliability_means = np.zeros((len(num_users), 3))
    colors = ['g', 'r', 'b']
    linestyles = ['-', '--', '-.']

    _controller = GlobecomController(q_manager_id=q_manager_id)
    sinr_levels = _controller.sinr_levels
    capacity_levels = _controller.capacity_levels

    total_rewards = [[] for i in range(len(num_users))]
    total_fairnesses = [[] for i in range(len(num_users))]
    total_reliabilities = [[] for i in range(len(num_users))]

    total_sinr_stats = [[] for i in range(len(num_users))]
    total_capacity_stats = [[] for i in range(len(num_users))]
    total_sinr_means = [[] for i in range(len(num_users))]
    total_capacity_means = [[] for i in range(len(num_users))]
    energies_spent = np.zeros((len(num_users), 3))
    for n_users_idx, n_users in enumerate(num_users):
        # Alternating locations run
        rewards_total = np.zeros((N_CYCLES,))
        fairness_total = np.zeros((N_CYCLES,))
        reliability_total = np.zeros((N_CYCLES,))
        sinr_stats = np.zeros((3, sinr_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_stats = np.zeros((3, capacity_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_means = []
        sinr_means = []
        for iteration_idx in tqdm(range(1, int(N_ITERS) + 1)):
            _controller = GlobecomController(q_manager_id=q_manager_id, n_users=n_users, speed_divisor=speed_divisors[iteration_idx - 1])
            _controller.reset_model(n_users)
            for cycle_idx in range(N_CYCLES):
                cycle_reward, cycle_fairness, cycle_reliability = _controller.run_cycle_with_fixed_locations(
                    alternating=True)
                rewards_total[cycle_idx] = rewards_total[cycle_idx] + (
                        cycle_reward - rewards_total[cycle_idx]) / iteration_idx
                fairness_total[cycle_idx] = fairness_total[cycle_idx] + (
                        cycle_fairness - fairness_total[cycle_idx]) / iteration_idx
                reliability_total[cycle_idx] = reliability_total[cycle_idx] + (
                        cycle_reliability - reliability_total[cycle_idx]) / iteration_idx
            sinr_stats += _controller.sinr_stats
            capacity_stats += _controller.capacity_stats
            sinr_mean, capacity_mean = _controller.simulation_controller.get_users_rf_means()
            capacity_means.append(capacity_mean)
            sinr_means.append(sinr_mean)
            energies_spent[n_users_idx][0] += _controller.simulation_controller.get_uavs_total_consumed_energy()

        total_rewards[n_users_idx].append(rewards_total)
        total_fairnesses[n_users_idx].append(fairness_total)
        total_reliabilities[n_users_idx].append(reliability_total)
        energies_spent[n_users_idx][0] /= N_ITERS
        total_capacity_means[n_users_idx].append(np.array(capacity_means).flatten())
        total_sinr_means[n_users_idx].append(np.array(sinr_means).flatten())
        total_sinr_stats[n_users_idx].append(sinr_stats)
        total_capacity_stats[n_users_idx].append(capacity_stats)
        total_rewards_means[n_users_idx][0] = rewards_total.mean()
        total_fairness_means[n_users_idx][0] = fairness_total.mean()
        total_reliability_means[n_users_idx][0] = reliability_total.mean()
        np.save(results_folder + f'rewards_alternating_{n_users}', rewards_total)
        np.save(results_folder + f'fairness_alternating_{n_users}', fairness_total)
        np.save(results_folder + f'reliability_alternating_{n_users}', reliability_total)

        # Fixed locations run
        rewards_total = np.zeros((N_CYCLES,))
        fairness_total = np.zeros((N_CYCLES,))
        reliability_total = np.zeros((N_CYCLES,))
        sinr_stats = np.zeros((3, sinr_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_stats = np.zeros((3, capacity_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_means = []
        sinr_means = []
        for iteration_idx in tqdm(range(1, int(N_ITERS) + 1)):
            _controller = GlobecomController(q_manager_id=q_manager_id, n_users=n_users, speed_divisor=speed_divisors[iteration_idx - 1])
            _controller.reset_model(n_users)
            for cycle_idx in range(N_CYCLES):
                cycle_reward, cycle_fairness, cycle_reliability = _controller.run_cycle_with_fixed_locations(
                    alternating=False)
                rewards_total[cycle_idx] = rewards_total[cycle_idx] + (
                        cycle_reward - rewards_total[cycle_idx]) / iteration_idx
                fairness_total[cycle_idx] = fairness_total[cycle_idx] + (
                        cycle_fairness - fairness_total[cycle_idx]) / iteration_idx
                reliability_total[cycle_idx] = reliability_total[cycle_idx] + (
                        cycle_reliability - reliability_total[cycle_idx]) / iteration_idx
            sinr_stats += _controller.sinr_stats
            capacity_stats += _controller.capacity_stats
            sinr_mean, capacity_mean = _controller.simulation_controller.get_users_rf_means()
            capacity_means.append(capacity_mean)
            sinr_means.append(sinr_mean)
            energies_spent[n_users_idx][1] += _controller.simulation_controller.get_uavs_total_consumed_energy()

        total_rewards[n_users_idx].append(rewards_total)
        total_fairnesses[n_users_idx].append(fairness_total)
        total_reliabilities[n_users_idx].append(reliability_total)
        energies_spent[n_users_idx][1] /= N_ITERS
        total_capacity_means[n_users_idx].append(np.array(capacity_means).flatten())
        total_sinr_means[n_users_idx].append(np.array(sinr_means).flatten())
        total_sinr_stats[n_users_idx].append(sinr_stats)
        total_capacity_stats[n_users_idx].append(capacity_stats)
        total_rewards_means[n_users_idx][1] = rewards_total.mean()
        total_fairness_means[n_users_idx][1] = fairness_total.mean()
        total_reliability_means[n_users_idx][1] = reliability_total.mean()
        np.save(results_folder + f'rewards_fixed_{n_users}', rewards_total)
        np.save(results_folder + f'fairness_fixed_{n_users}', fairness_total)
        np.save(results_folder + f'reliability_fixed_{n_users}', reliability_total)

        # Q Learning
        rewards_total = np.zeros((N_CYCLES,))
        fairness_total = np.zeros((N_CYCLES,))
        reliability_total = np.zeros((N_CYCLES,))
        sinr_stats = np.zeros((3, sinr_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_stats = np.zeros((3, capacity_levels.shape[0] - 1), dtype=int)  # mean, max, min stats
        capacity_means = []
        sinr_means = []
        for iteration_idx in tqdm(range(1, N_ITERS + 1)):
            _controller = GlobecomController(q_manager_id=q_manager_id, n_users=n_users, speed_divisor=speed_divisors[iteration_idx - 1])
            _controller.reset_model(n_users)
            rewards_total_iter, fairness_total_iter, reliability_total_iter \
                = _controller.run_n_cycles(N_CYCLES)
            rewards_total = rewards_total + (rewards_total_iter - rewards_total) / iteration_idx
            fairness_total = fairness_total + (fairness_total_iter - fairness_total) / iteration_idx
            reliability_total = reliability_total + (reliability_total_iter - reliability_total) / iteration_idx
            sinr_stats += _controller.sinr_stats
            capacity_stats += _controller.capacity_stats
            sinr_mean, capacity_mean = _controller.simulation_controller.get_users_rf_means()
            capacity_means.append(capacity_mean)
            sinr_means.append(sinr_mean)
            energies_spent[n_users_idx][2] += _controller.simulation_controller.get_uavs_total_consumed_energy()

        total_rewards[n_users_idx].append(rewards_total)
        total_fairnesses[n_users_idx].append(fairness_total)
        total_reliabilities[n_users_idx].append(reliability_total)
        energies_spent[n_users_idx][2] /= N_ITERS
        total_capacity_means[n_users_idx].append(np.array(capacity_means).flatten())
        total_sinr_means[n_users_idx].append(np.array(sinr_means).flatten())
        total_sinr_stats[n_users_idx].append(sinr_stats)
        total_capacity_stats[n_users_idx].append(capacity_stats)
        total_rewards_means[n_users_idx][2] = rewards_total.mean()
        total_fairness_means[n_users_idx][2] = fairness_total.mean()
        total_reliability_means[n_users_idx][2] = reliability_total.mean()
        np.save(results_folder + f'rewards_q_{n_users}', rewards_total)
        np.save(results_folder + f'fairness_q_{n_users}', fairness_total)
        np.save(results_folder + f'reliability_q_{n_users}', reliability_total)

    np.save(results_folder + f'rewards_means_{n_users}', total_rewards_means)
    np.save(results_folder + f'fairness_means_{n_users}', total_fairness_means)
    np.save(results_folder + f'reliability_means_{n_users}', total_reliability_means)

    rewards_fig, rewards_axs = plt.subplots(3, len(num_users), sharex=True, sharey='row')
    if len(num_users) == 1:
        rewards_axs = [[rw_ax] for rw_ax in rewards_axs]
    min_y_rewards = max(min([total_rewards[i][j].min() for i in range(len(total_rewards)) for j in range(len(total_rewards[i]))]), 0)
    min_y_fairnesses = max(min([total_fairnesses[i][j].min() for i in range(len(total_fairnesses)) for j in range(len(total_fairnesses[i]))]), 0)
    min_y_reliabilitites = max(min([total_reliabilities[i][j].min() for i in range(len(total_reliabilities)) for j in range(len(total_reliabilities[i]))]), 0)
    max_y_rewards = max([total_rewards[i][j].max() for i in range(len(total_rewards)) for j in range(len(total_rewards[i]))])
    max_y_fairnesses = max([total_fairnesses[i][j].max() for i in range(len(total_fairnesses)) for j in range(len(total_fairnesses[i]))])
    max_y_reliabilitites = max([total_reliabilities[i][j].max() for i in range(len(total_reliabilities)) for j in range(len(total_reliabilities[i]))])
    rewards_axs[0][0].set_ylabel('Reward', fontsize=10)
    rewards_axs[1][0].set_ylabel('Fairness Score', fontsize=10)
    rewards_axs[2][0].set_ylabel('Reliability Score', fontsize=10)
    # rewards_axs[0][0].set_yticks(np.arange(min_y_rewards, max_y_rewards + 2 * (max_y_rewards - min_y_rewards)/5, 7, dtype=int) )
    # rewards_axs[1][0].set_yticks(np.around(np.linspace(min_y_fairnesses, max_y_fairnesses + 1 * (max_y_fairnesses - min_y_fairnesses)/5, 7, dtype=float), 2 ))
    # rewards_axs[2][0].set_yticks(np.around(np.linspace(min_y_reliabilitites, max_y_reliabilitites + 2 * (max_y_reliabilitites - min_y_reliabilitites)/5, 7, dtype=float),2 ))
    # rewards_axs[0][0].set_ylim([min_y_rewards, max_y_rewards + (max_y_rewards - min_y_rewards)/10])
    # rewards_axs[1][0].set_ylim([min_y_fairnesses, max_y_fairnesses])
    # rewards_axs[2][0].set_ylim([min_y_reliabilitites, max_y_reliabilitites])
    rewards_axs[2][0].set_xlim([0, 235])
    rewards_axs[2][1].set_xlim([0, 235])
    rewards_axs[2][2].set_xlim([0, 235])
    for n_users_idx, n_users in enumerate(num_users):
        rewards_axs[0][n_users_idx].plot(total_rewards[n_users_idx][0], label='Alternating', color=colors[2], ls=linestyles[2])
        rewards_axs[1][n_users_idx].plot(total_fairnesses[n_users_idx][0], label='Alternating', color=colors[2], ls=linestyles[2])
        rewards_axs[2][n_users_idx].plot(total_reliabilities[n_users_idx][0], label='Alternating', color=colors[2], ls=linestyles[2])
        rewards_axs[0][n_users_idx].plot(total_rewards[n_users_idx][1], label='Fixed', color=colors[1], ls=linestyles[1])
        rewards_axs[1][n_users_idx].plot(total_fairnesses[n_users_idx][1], label='Fixed', color=colors[1], ls=linestyles[1])
        rewards_axs[2][n_users_idx].plot(total_reliabilities[n_users_idx][1], label='Fixed', color=colors[1], ls=linestyles[1])
        rewards_axs[0][n_users_idx].plot(total_rewards[n_users_idx][2], label=q_manager_id, color=colors[0], ls=linestyles[0])
        rewards_axs[1][n_users_idx].plot(total_fairnesses[n_users_idx][2], label=q_manager_id, color=colors[0], ls=linestyles[0])
        rewards_axs[2][n_users_idx].plot(total_reliabilities[n_users_idx][2], label=q_manager_id, color=colors[0], ls=linestyles[0])
        rewards_axs[2][n_users_idx].set_xlabel(f'Cycle\n{n_users} users', fontsize=10)
        rewards_axs[2][n_users_idx].set_xticks(np.arange(0, max(2, 235), 50))
    total_handles = []
    total_labels = []
    for i in range(len(rewards_axs)):
        handles, labels = rewards_axs[i][n_users_idx].get_legend_handles_labels()
        total_handles.append(handles)
        total_labels.append(labels)
    # rewards_fig.legend(handles, labels, loc='upper center')
    rewards_fig.tight_layout()
    rewards_fig.align_ylabels()
    rewards_fig.legend(handles, labels, loc='upper center', ncol=3)
    rewards_fig.subplots_adjust(left=0.1, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.1)
    rewards_fig.show()
    rewards_fig.savefig(results_folder + 'results_subplots.eps', format='eps')
    rewards_fig.savefig(results_folder + 'results_subplots.png', format='png')

    means_fig, means_axs = plt.subplots(3, len(num_users), sharex='col', sharey='row')
    if len(num_users) == 1:
        means_axs = [[rw_ax] for rw_ax in means_axs]
    min_y_rewards = max(min([total_rewards_means[i][j].min() for i in range(len(total_rewards_means)) for j in range(len(total_rewards_means[i]))]), 0)
    min_y_fairnesses = max(min([total_fairness_means[i][j].min() for i in range(len(total_fairness_means)) for j in range(len(total_fairness_means[i]))]), 0)
    min_y_reliabilitites = max(min([total_reliability_means[i][j].min() for i in range(len(total_reliability_means)) for j in range(len(total_reliability_means[i]))]), 0)
    max_y_rewards = max([total_rewards_means[i][j].max() for i in range(len(total_rewards_means)) for j in range(len(total_rewards_means[i]))])
    max_y_fairnesses = max([total_fairness_means[i][j].max() for i in range(len(total_fairness_means)) for j in range(len(total_fairness_means[i]))])
    max_y_reliabilitites = max([total_reliability_means[i][j].max() for i in range(len(total_reliability_means)) for j in range(len(total_reliability_means[i]))])
    means_axs[0][0].set_ylabel('Mean Reward', fontsize=10)
    means_axs[1][0].set_ylabel('Mean Fairness', fontsize=10)
    means_axs[2][0].set_ylabel('Mean Reliability', fontsize=10)
    means_axs[0][0].set_ylim([min_y_rewards, max_y_rewards])
    means_axs[1][0].set_ylim([min_y_fairnesses, max_y_fairnesses])
    means_axs[2][0].set_ylim([min_y_reliabilitites, max_y_reliabilitites])
    means_axs[0][0].set_yticks(np.arange(min_y_rewards, max_y_rewards + (max_y_rewards - min_y_rewards)/5, (max_y_rewards - min_y_rewards)/5, dtype=int) )
    means_axs[1][0].set_yticks(np.around(np.arange(min_y_fairnesses - (max_y_fairnesses - min_y_fairnesses)/5, max_y_fairnesses + (max_y_fairnesses - min_y_fairnesses)/5, (max_y_fairnesses - min_y_fairnesses)/5, dtype=float), 2 )[:7])
    means_axs[2][0].set_yticks(np.around(np.arange(max(min_y_reliabilitites - (max_y_reliabilitites - min_y_reliabilitites)/5, 0), max_y_reliabilitites + (max_y_reliabilitites - min_y_reliabilitites)/5, (max_y_reliabilitites - min_y_reliabilitites)/5, dtype=float),2 ))
    X = ['Alternating', 'Fixed'] + [f'{q_manager_id}']
    width = 0.025
    X_axis = np.arange(width, width*(len(X)), width)
    # means_axs[2].set_xticks([width*(len(X) + 1)/2], [f'{NUM_OF_USERS} users'])
    shift = (len(num_users) * 3 * width) + 2 * width
    # X_axis = np.array([np.arange(width, width * (len(X)), width) + i * shift for i in range(len(num_users))]).flatten()
    for n_users_idx, n_users in enumerate(num_users):
        means_axs[2][n_users_idx].set_xticks(X_axis, ['',f'{n_users} users',''])
        for idx, _mean in enumerate(total_rewards_means[n_users_idx]):
            means_axs[0][n_users_idx].bar(X_axis[idx], _mean, label=X[idx], width=width, color=colors[2 - idx], ls=linestyles[2 - idx])
        for idx, _mean in enumerate(total_fairness_means[n_users_idx]):
            means_axs[1][n_users_idx].bar(X_axis[idx], _mean, label=X[idx], width=width, color=colors[2 - idx], ls=linestyles[2 - idx])
        for idx, _mean in enumerate(total_reliability_means[n_users_idx]):
            means_axs[2][n_users_idx].bar(X_axis[idx], _mean, label=X[idx], width=width, color=colors[2 - idx], ls=linestyles[2 - idx])
    means_fig.tight_layout()
    for i in range(len(means_axs)):
        handles, labels = means_axs[i][len(num_users) - 1].get_legend_handles_labels()
        total_handles.append(handles)
        total_labels.append(labels)
    means_fig.align_ylabels()
    means_fig.legend(handles, labels, loc='upper center', ncol=3)
    means_fig.subplots_adjust(left=0.1, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.1)
    means_fig.show()
    means_fig.savefig(results_folder + 'means_subplots.eps', format='eps')
    means_fig.savefig(results_folder + 'means_subplots.png', format='png')

    def ecdf_x_y(a):
        def ecdf(a):
            x, counts = np.unique(a, return_counts=True)
            cusum = np.cumsum(counts)
            return x, cusum / cusum[-1]

        x, y = ecdf(a)
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.)
        return x, y

    rf_means_fig, rf_means_axs = plt.subplots(len(num_users), 2, sharex='col', sharey='row')
    if len(num_users) == 1:
        rf_means_axs = [rf_means_axs]
    rf_means_axs[len(num_users) - 1][0].set_xlabel('Mean SINR [dB]', fontsize=10)
    rf_means_axs[len(num_users)- 1][1].set_xlabel('Mean Throughput [Mbps]', fontsize=10)
    X = ['Alternating', 'Fixed'] + [f'{q_manager_id}']
    for n_users_idx, n_users in enumerate(num_users):
        rf_means_axs[n_users_idx][0].set_ylabel(f'{num_users[n_users_idx]} users\nCDF')
        for sinr_mean, label, idx in zip(total_sinr_means[n_users_idx], X, [2,1,0]):
            x, y = ecdf_x_y(sinr_mean)
            rf_means_axs[n_users_idx][0].plot(x, y, drawstyle='steps-post', label=label, color=colors[idx], ls=linestyles[idx], alpha=0.7)
            rf_means_axs[n_users_idx][0].grid(True)
        # rf_means_axs[n_users_idx][0].legend()
        for capacity_mean, label, idx in zip(total_capacity_means[n_users_idx], X, [2, 1, 0]):
            x, y = ecdf_x_y(capacity_mean)
            rf_means_axs[n_users_idx][1].plot(x, y, drawstyle='steps-post', label=label, color=colors[idx],
                                              ls=linestyles[idx], alpha=0.7)
            rf_means_axs[n_users_idx][1].grid(True)
            plt.grid(True)
    rf_means_axs[n_users_idx][1].legend()
    rf_means_fig.tight_layout()
    rf_means_fig.align_ylabels()
    rf_means_fig.subplots_adjust(left=0.15, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.05)
    rf_means_fig.savefig(results_folder + 'sinr_throuhput_means_cdf.eps', format='eps')
    rf_means_fig.savefig(results_folder + 'sinr_throuhput_means_cdf.png', format='png')
    rf_means_fig.show()

    sinr_stats_fig, sinr_stats_axs = plt.subplots(3, len(num_users), sharey='row', sharex='col')
    if len(num_users) == 1:
        sinr_stats_axs = [[rw_ax] for rw_ax in sinr_stats_axs]
    sinr_stats_axs[0][0].set_ylabel('SINR [dB]', fontsize=10)
    sinr_stats_axs[1][0].set_ylabel('Min SINR [dB]', fontsize=10)
    sinr_stats_axs[2][0].set_ylabel('Max SINR [dB]', fontsize=10)
    for n_users_idx, n_users in enumerate(num_users):
        data = []
        for idx in range(len(X)):
            data.append(np.repeat(sinr_levels[:-1], total_sinr_stats[n_users_idx][idx][0]))
        sinr_stats_axs[0][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)
        sinr_stats_axs[0][n_users_idx].set_title(f'{num_users[n_users_idx]} users')

        data = []
        for idx in range(len(X)):
            data.append(np.repeat(sinr_levels[:-1], total_sinr_stats[n_users_idx][idx][1]))
        sinr_stats_axs[1][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)

        data = []
        for idx in range(len(X)):
            data.append(np.repeat(sinr_levels[:-1], total_sinr_stats[n_users_idx][idx][2]))
        sinr_stats_axs[2][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)

    sinr_stats_axs[0][n_users_idx].set_yticklabels(X * 3, rotation=45)
    sinr_stats_axs[1][n_users_idx].set_yticklabels(X* 3, rotation=45)
    sinr_stats_axs[2][n_users_idx].set_yticklabels(X* 3, rotation=45)
    sinr_stats_fig.tight_layout()
    sinr_stats_fig.align_ylabels()
    sinr_stats_fig.subplots_adjust(left=0.20, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.05)
    sinr_stats_fig.savefig(results_folder + 'sinr_stats_boxplots.eps', format='eps')
    sinr_stats_fig.savefig(results_folder + 'sinr_stats_boxplots.png', format='png')
    sinr_stats_fig.show()

    capacity_stats_fig, capacity_stats_axs = plt.subplots(3, len(num_users), sharey='row', sharex='col')
    if len(num_users) == 1:
        capacity_stats_axs = [[rw_ax] for rw_ax in capacity_stats_axs]
    capacity_stats_axs[0][0].set_ylabel('Throuhput\n[Kbps]', fontsize=10)
    capacity_stats_axs[1][0].set_ylabel('Min Throuhput\n[Kbps]', fontsize=10)
    capacity_stats_axs[2][0].set_ylabel('Max Throuhput\n[Mbps]', fontsize=10)
    for n_users_idx, n_users in enumerate(num_users):
        data = []
        for idx in range(len(X)):
            data.append(np.repeat(capacity_levels[:-1], total_capacity_stats[n_users_idx][idx][0]))
        capacity_stats_axs[0][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)
        capacity_stats_axs[0][n_users_idx].set_title(f'{num_users[n_users_idx]} users')

        data = []
        for idx in range(len(X)):
            data.append(np.repeat(capacity_levels[:-1], total_capacity_stats[n_users_idx][idx][1]))
        capacity_stats_axs[1][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)

        data = []
        for idx in range(len(X)):
            data.append(np.repeat(capacity_levels[:-1] * 1024, total_capacity_stats[n_users_idx][idx][2])/1024)
        capacity_stats_axs[2][n_users_idx].boxplot(data, patch_artist=True, vert=False, showfliers=False)
    capacity_stats_axs[0][n_users_idx].set_yticklabels(X * 3, rotation=45)
    capacity_stats_axs[1][n_users_idx].set_yticklabels(X*3, rotation=45)
    capacity_stats_axs[2][n_users_idx].set_yticklabels(X*3, rotation=45)
    capacity_stats_fig.tight_layout()
    capacity_stats_fig.align_ylabels()
    capacity_stats_fig.subplots_adjust(left=0.2, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.05)
    capacity_stats_fig.savefig(results_folder + 'capacity_stats_boxplots.eps', format='eps')
    capacity_stats_fig.savefig(results_folder + 'capacity_stats_boxplots.png', format='png')
    capacity_stats_fig.show()


    width = 0.025
    energy_fig, energy_axs = plt.subplots(1, len(num_users), sharey='row')
    if len(num_users) == 1:
        energy_axs = [energy_axs]
    energy_axs[0].set_ylabel('Energy [W]')
    for n_users_idx, n_users in enumerate(num_users):
        # energy_axs[n_users_idx].bar([1,2,3], energies_spent[n_users_idx])
        # for idx, energy in enumerate(energies_spent):
        energy_axs[n_users_idx].set_title(f'{n_users} users')
        energy_axs[n_users_idx].bar(X_axis, energies_spent[n_users_idx], width=width, label=X, color=colors[::-1])
        energy_axs[n_users_idx].set_xticks(X_axis, labels=X, rotation=45)
    energy_fig.tight_layout()
    energy_fig.align_ylabels()
    energy_fig.subplots_adjust(left=0.15, top=0.93, right=0.95, bottom=0.15, wspace=0.05, hspace=0.05)
    energy_fig.savefig(results_folder + 'energies.eps', format='eps')
    energy_fig.savefig(results_folder + 'energies.png', format='png')
    energy_fig.show()

# for n_users_idx, n_users in enumerate(num_users):
#     if n_users_idx == 0:
#         continue
#     for i in range(len(total_reliabilities)):
#         total_reliabilities[n_users_idx][i] = 1 - ((1 - total_reliabilities[n_users_idx][i]) * 150)/n_users