import numpy as np

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    tx_coords = MBS_LOCATION
    bw_rs = np.linspace(0.25e-3, 0.25, 1000)
    res = []
    for i in range(NUM_UAVS):
        max_gains = []
        best_bw_radius = []
        for loc_2d in UAVS_LOCATIONS[i]:
            for height in UAVS_HEIGHTS:
                rx_coords = Coords3d(loc_2d[0], loc_2d[1], height)
                max_gain = 0
                best_bw = bw_rs[0]
                for bw_r in bw_rs:
                    gain, _,_,_,_ =StatisticalModel.get_charge_power_and_capacity(tx_coords, rx_coords, get_gain=True, beamwaist_radius=bw_r)
                    gain = db2lin(gain)
                    if gain > max_gain:
                        max_gain = gain
                        best_bw = bw_r
                max_gains.append(max_gain)
                best_bw_radius.append(best_bw)
        res.append([max_gains, best_bw_radius])

    means = []
    for _res in res:
        bws = np.array(_res[1])
        means.append(bws.mean())

