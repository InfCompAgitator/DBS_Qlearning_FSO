import numpy as np

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    tx_coords = MBS_LOCATION

    step = 5
    distance = np.linspace(step, step * 100, 100)
    distance = np.sqrt(distance**2 + distance**2 + distance**2)
    gains = []
    capacities = []
    rx_coords = MBS_LOCATION.copy()
    for i in range(100):
        rx_coords+=step
        gain, _,_,_,_ =StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=True)
        gain = db2lin(gain)
        gains.append(gain)
        _, capacity = StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, transmit_power=0.2, power_split_ratio=1)
        capacities.append(capacity/(1024*1024))

    plt.plot(distance, gains, label='non-adaptive beamwidth')
    plt.legend(loc='lower left')
    ax = plt.gca()
    ax.set_xlabel('Distance [m]', fontsize=15, **csfont)
    ax.set_ylabel('Link gain [linear]', fontsize=15, **csfont)
    # plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/distance_vs_fso_gain_fixed_bw.eps', format='eps')
    plt.show()

    plt.plot(distance, capacities, label='non-adaptive beamwidth')
    plt.legend(loc='lower left')
    ax = plt.gca()
    ax.set_xlabel('Distance [m]', fontsize=15, **csfont)
    ax.set_ylabel('Capacity [mbps]', fontsize=15, **csfont)
    # plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/distance_vs_fso_gain_fixed_bw.eps', format='eps')
    plt.show()

