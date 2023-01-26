from src.channel_model.rf_a2g import PlosModel
import matplotlib.pyplot as plt
import numpy as np
from src.environment.obstacles import *
from src.parameters import *
from scipy import stats


if __name__ == '__main__':
    obs = get_madrid_buildings()
    total_xs = []
    total_ys = []
    areas = []
    heights = []
    for _obs in obs.obstaclesList:
        xs = _obs.vertices[0][0], _obs.vertices[2][0]
        ys = _obs.vertices[0][1], _obs.vertices[1][1]
        total_xs+=xs
        total_ys+=ys
        heights.append(_obs.height)
        areas.append(abs(xs[0]-xs[1]) * abs(ys[0]-ys[1]))


    max_x, max_y = max(total_xs)+ BUILDINGS_AREA_MARGIN, max(total_ys)+ BUILDINGS_AREA_MARGIN

    beta = len(obs.obstaclesList)/ (max_x * max_y *1e-6)
    alpha = sum(areas) / (max_x * max_y)

    loc, gamma = stats.rayleigh.fit(heights)

    PlosModel.get_a_b_params(alpha=alpha, beta=beta, gamma=gamma)