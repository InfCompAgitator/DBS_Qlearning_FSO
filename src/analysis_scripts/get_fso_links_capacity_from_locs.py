import numpy as np

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib.pyplot as plt
import itertools
from os.path import dirname, pardir, join as pjoin
import scipy.io as sio
from src.data_structures import Coords3d
from matplotlib import patches
import matplotlib as mpl

if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}
    fig = plt.figure()
    loc_file = pjoin(pardir, pardir, 'junk', 'dbs_locations_10000_40.mat')
    coords = sio.loadmat(loc_file)
    coords = coords['mu']

    cap_file = pjoin(pardir, pardir, 'junk', 'capacities_10000_40.mat')
    capacs = sio.loadmat(cap_file)
    capacs = capacs['capacity_per_drone'][0]

    poss_links = list(itertools.product(range(coords.__len__()),
                           range(coords.__len__())))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for poss_link in poss_links:
        if (poss_link[0] == poss_link[1]):
            continue
        i,j = poss_link[0], poss_link[1]
        dbs_1 = Coords3d(coords[i][0], coords[i][1], 0)
        dbs_2 = Coords3d(coords[j][0], coords[j][1], 0)
        if dbs_1.get_distance_to(dbs_2) > 2000:
            continue
        _, capacity = StatisticalModel.get_charge_power_and_capacity(dbs_1, dbs_2, transmit_power=0.2, power_split_ratio=1)
        capacity = int(capacity/(1024*1024))
        ax.plot([dbs_1.x, dbs_2.x], [dbs_1.y, dbs_2.y],
                     color='black', linestyle='dashed', linewidth=0.5)
        ax.text(abs(dbs_1.x+dbs_2.x)/2, abs(dbs_1.y+dbs_2.y)/2, f'{capacity}', fontsize=6)
    fig.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(capacs.size):
        dbs_1 = Coords3d(coords[i][0], coords[i][1], 0)
        circle = patches.Circle((dbs_1.x+400, dbs_1.y+200), radius=500, color='yellow')
        ax.add_patch(circle)
        ax.plot(dbs_1.x, dbs_1.y)
        ax.text(dbs_1.x, dbs_1.y, f'{capacs[i]}', fontsize=12)
    fig.show()

    # coords =