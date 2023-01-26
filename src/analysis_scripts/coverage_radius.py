from src.channel_model.rf_a2g import PlosModel
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Plotting R vs H
    rs = []
    max_height = 1000
    for uav_height in range(20, max_height, 10):
        rs.append(PlosModel.get_coverage_radius(uav_height))
    plt.plot(range(20, max_height, 10), rs)
    plt.show()
