from src.main_controller import SimulationController
from time import sleep
from src.parameters import *
from src.channel_model.fso_a2a import StatisticalModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    # a = SimulationController()
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    tx_coords = Coords3d(0, 0, MBS_HEIGHT)
    rx_coords = Coords3d(1, 0, MBS_HEIGHT)
    gains1 = []
    StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=False, fixed_bw=1,
                                                   fixed_orientation=True, transmit_power=600, power_split_ratio=0.001)
    xs = np.linspace(10, 1000, 500)
    for idx, x in enumerate(xs):
        rx_coords.x = x
        gains1.append(StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=True, fixed_bw=1, fixed_orientation=True)[0])
    plt.plot(xs, gains1, label='Adaptive beamwidth and lens plane')

    gains2 = []
    for idx, x in enumerate(xs):
        rx_coords.x = x
        gains2.append(StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=True, fixed_bw=False,
                                                                    fixed_orientation=True)[0])
    plt.plot(xs, gains2, label='Adaptive beamwidth, non-adaptive lens')

    gains3 = []
    for idx, x in enumerate(xs):
        rx_coords.x = x
        gains3.append(StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=True, fixed_bw=False,
                                                                    fixed_orientation=False)[0])
    plt.plot(xs, gains3, label='Non-adaptive beamwidth and lens plane')
    plt.legend(loc='lower left')
    ax = plt.gca()
    ax.set_xlabel('x [m]', fontsize=15, **csfont)
    ax.set_ylabel('Link gain [dB]', fontsize=15, **csfont)
    # plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/charging_gain.eps', format='eps')
    plt.show()