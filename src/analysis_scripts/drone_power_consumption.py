import numpy as np
from varname import nameof
from src.drone_station import DroneStation
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.parameters import *
import itertools

if __name__ == '__main__':
    uav_1 = DroneStation()
    # uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=1)

    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    p_hover = uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=0)
    p_upward = uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=1)
    p_downward = uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=-1)
    p_horizontal = uav_1.battery.get_consumption_power(speed_x=1, speed_y=0, speed_z=0)
    p_45_up = uav_1.battery.get_consumption_power(speed_x=1, speed_y=0, speed_z=1)
    p_45_down = uav_1.battery.get_consumption_power(speed_x=1, speed_y=0, speed_z=-1)

    speed = np.linspace(1, 30, 30)
    vf = np.vectorize(uav_1.battery.get_consumption_power)
    ps_hover =vf(speed_x=0, speed_y=0, speed_z=0*speed)
    ps_upward = vf(speed_x=0, speed_y=0, speed_z=speed)
    ps_downward = vf(speed_x=0, speed_y=0, speed_z=-speed)
    ps_horizontal = vf(speed_x=speed, speed_y=0, speed_z=0)
    # ps_45_up = vf(speed_x=speed/np.sqrt(2), speed_y=0, speed_z=speed/np.sqrt(2))
    # ps_45_down = vf(speed_x=speed/np.sqrt(2), speed_y=0, speed_z=-speed/np.sqrt(2))
    ps_45_up = vf(speed_x=speed, speed_y=0, speed_z=speed)
    ps_45_down = vf(speed_x=speed, speed_y=0, speed_z=-speed)

    ys = [ps_hover, ps_upward, ps_downward, ps_horizontal, ps_45_up, ps_45_down]
    labels = ["hover", "upward", "downward", "horizontal", "45 degrees up", "45 degrees down"]
    ys = [ps_hover,  ps_horizontal]
    labels = ["Hovering", "Horizontal flight"]
    for idx, y in enumerate(ys):
        plt.plot(speed, y, label=labels[idx])

    ax = plt.gca()
    ax.set_xlabel('Speed [m/s]', fontsize=15, **csfont)
    ax.set_ylabel('Consumed power [W]', fontsize=15, **csfont)
    plt.axvline(x=13, color='r', linestyle=':', label='Minimum consumption, x=13')
    plt.legend(loc='upper left', ncol=1, fontsize=12)
    plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/power_vs_speed.eps',
                format='eps')
    plt.show()

    # QLearningParams.TIME_STEP_Q = 25
    #
    # uav_id = 0
    # uav_locs = [Coords3d.from_array(UAVS_LOCATIONS[uav_id][0] + [UAVS_HEIGHTS[0]]),
    #             Coords3d.from_array(UAVS_LOCATIONS[uav_id][1] + [UAVS_HEIGHTS[0]]),]
    #             # Coords3d.from_array(UAVS_LOCATIONS[uav_id][0] + [UAVS_HEIGHTS[1]]),
    #             # Coords3d.from_array(UAVS_LOCATIONS[uav_id][1] + [UAVS_HEIGHTS[1]])]
    #
    # all_flights = list(itertools.permutations(uav_locs, 2))
    #
    # distances = []
    # for _flight in all_flights:
    #     distances.append(_flight[0].get_distance_to(_flight[1]))
    #
    # powers = []
    # durations = []
    # for idx, _distance in enumerate(distances):
    #     # idx = distances.index(_distance)
    #     travel_duration = _distance / UAV_TRAVEL_SPEED
    #     durations.append(travel_duration)
    #     speeds = (all_flights[idx][0] - all_flights[idx][1])/ travel_duration
    #     powers.append(uav_1.battery.get_consumption_power(speed_x=speeds.x, speed_y=speeds.y, speed_z=speeds.z))
    #
    #
    # travel_energies = np.array(powers) * np.array(durations)
    # hovering_durations = QLearningParams.TIME_STEP_Q - np.array(durations)
    # hovering_energies = hovering_durations * uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=0)
    #
    # energy_consumptions = np.append(travel_energies + hovering_energies,
    #                                 uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=0) * QLearningParams.TIME_STEP_Q)
    #
    # horizontal_energy = energy_consumptions[0]
    # downward_energy = energy_consumptions[1]
    # downward_45_energy = energy_consumptions[2]
    # upward_energy = energy_consumptions[6]
    # upward_45_energy = energy_consumptions[7]
    # hovering_energy = QLearningParams.TIME_STEP_Q * uav_1.battery.get_consumption_power(speed_x=0, speed_y=0, speed_z=0)






