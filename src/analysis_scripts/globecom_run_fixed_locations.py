from src.globecom_controller import GlobecomController, N_CYCLES
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_results():
    pass


if __name__ == '__main__':
    q_managers_ids = ['F3']  # 00', '01', '02', '03', 'H0', 'H1', 'H2', 'H3']
    rewards_fig, rewards_axs = plt.subplots(3, 1, sharex=True)
    rewards_axs[0].set_ylabel('Reward', fontsize=10)
    rewards_axs[1].set_ylabel('Fairness Score', fontsize=10)
    rewards_axs[2].set_ylabel('Reliability Score', fontsize=10)

    total_rewards_means = []
    total_fairness_means = []
    total_reliability_means = []

    _controller = GlobecomController(q_manager_id=q_managers_ids[0])
    rewards_total = np.zeros((N_CYCLES,))
    fairness_total = np.zeros((N_CYCLES,))
    reliability_total = np.zeros((N_CYCLES,))
    for _cycle in tqdm(range(N_CYCLES)):
        rewards_total[_cycle], fairness_total[_cycle], reliability_total[
            _cycle] = _controller.run_cycle_with_fixed_locations(alternating=True)
    total_rewards_means.append(rewards_total.mean())
    total_fairness_means.append(fairness_total.mean())
    total_reliability_means.append(reliability_total.mean())
    rewards_axs[0].plot(rewards_total, label='Alternating')
    rewards_axs[1].plot(fairness_total, label='Alternating')
    rewards_axs[2].plot(reliability_total, label='Alternating')

    _controller = GlobecomController(q_manager_id=q_managers_ids[0])
    rewards_total = np.zeros((N_CYCLES,))
    for _cycle in tqdm(range(N_CYCLES)):
        rewards_total[_cycle], fairness_total[_cycle], reliability_total[
            _cycle] = _controller.run_cycle_with_fixed_locations(alternating=False)
    total_rewards_means.append(rewards_total.mean())
    total_fairness_means.append(fairness_total.mean())
    total_reliability_means.append(reliability_total.mean())
    rewards_axs[0].plot(rewards_total, label='Fixed')
    rewards_axs[1].plot(fairness_total, label='Fixed')
    rewards_axs[2].plot(reliability_total, label='Fixed')

    for q_m_id in q_managers_ids:
        _controller = GlobecomController(q_manager_id=q_m_id)
        rewards_total = np.zeros((N_CYCLES,))
        rewards_total, fairness_total, reliability_total = _controller.run_n_cycles(N_CYCLES)
        total_rewards_means.append(rewards_total.mean())
        total_fairness_means.append(fairness_total.mean())
        total_reliability_means.append(reliability_total.mean())
        rewards_axs[0].plot(rewards_total, label=q_m_id)
        rewards_axs[1].plot(fairness_total, label=q_m_id)
        rewards_axs[2].plot(reliability_total, label=q_m_id)

        del _controller
    total_handles = []
    total_labels = []
    for i in range(len(rewards_axs)):
        handles, labels = rewards_axs[i].get_legend_handles_labels()
        total_handles.append(handles)
        total_labels.append(labels)
    # rewards_fig.legend(handles, labels, loc='upper center')
    rewards_fig.tight_layout()
    rewards_fig.legend(handles, labels, bbox_to_anchor=(1, 0), loc='lower right')
    rewards_fig.subplots_adjust(left=0.15, right=0.8, bottom=0.15, wspace=0.25, hspace=0.35)
    rewards_axs[2].set_xlabel('Cycle', fontsize=10)
    #
    rewards_fig.show()

    for _mean, label in zip(total_rewards_means, ['fixed'] + q_managers_ids):
        plt.bar(label, _mean)
