import numpy as np

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

if __name__ == '__main__':
    def get_req_power_and_gain(distance, beamwaist_radius, tx_coords, rx_coords):
        atmospheric_loss = StatisticalModel.get_atmospheric_loss(distance)
        atm_turb_induced_fading = 1  # for completeness
        responsivity = RX_RESPONSIVITY
        noise_power = NOISE_VARIANCE_FSO
        transmit_power = TX_POWER_FSO_MBS
        power_split_ratio = POWER_SPLIT_RATIO
        bandwidth = BANDWIDTH_FSO
        c_n = StatisticalModel.get_refraction_index(tx_coords, rx_coords)
        rho = StatisticalModel.get_coherence_length(c_n, wavelength, distance)
        beam_width = StatisticalModel.get_beamwidth(distance, rho, beamwaist_radius=beamwaist_radius)
        v1 = StatisticalModel.get_v1(beam_width)
        phi, theta = np.pi / 2, np.pi
        # phi, theta = StatisticalModel.get_orientation_angles(tx_coords,
        #                                                      rx_coords)
        v2 = StatisticalModel.get_v2(v1, phi, theta)
        t1 = StatisticalModel.get_t1(v1)
        t2 = StatisticalModel.get_t2(v2, phi, theta)
        t = (t1 + t2) / 2
        a0 = StatisticalModel.get_max_fraction_a0(v1, v2)
        lambda_1, lambda_2 = StatisticalModel.get_ig_fluctuations_eigen_values(tx_coords, rx_coords, phi, theta,
                                                                               distance,
                                                                               lens_radius)
        gml = StatisticalModel.get_gml(a0, t, beam_width, misalignment=np.sqrt(lambda_1 + lambda_2))
        gain = gml * atmospheric_loss * atm_turb_induced_fading * responsivity
        received_charge_power = (1 - power_split_ratio) * gain * transmit_power
        c = np.exp(1) / (2 * np.pi) * responsivity ** 2 * atmospheric_loss ** 2 * (
                power_split_ratio * transmit_power) ** 2 / noise_power
        r_max = 0.5 * np.log2(c * a0 ** 2)
        r_delta = 2 / (t * beam_width ** 2 * np.log(2)) * (lambda_1 + lambda_2)
        capacity = (r_max - r_delta) * bandwidth

        required_capacity = 20 * 1024 * 1024 * 40 / bandwidth  # 20Mb per symbol for 40 users

        required_power = 2 ** (2 * required_capacity + 4 * (lambda_1 + lambda_2) / (
                t * beam_width ** 2 * np.log(2))) / a0 * 2 * np.pi * noise_power / (
                                 np.exp(1) * responsivity ** 2 * atmospheric_loss ** 2)

        return required_power, gain


    wavelength = WAVELENGTH
    lens_radius = RX_DIAMETER / 2

    tx_coords = MBS_LOCATION
    i = 0
    itertools.permutations([UAVS_LOCATIONS[i][0], UAVS_HEIGHTS])
    res = []
    distances = []
    for i in range(NUM_UAVS):
        uav_req_power = []
        uav_gains = []
        d = []
        for loc_2d in UAVS_LOCATIONS[i]:
            for height in UAVS_HEIGHTS:
                rx_coords = Coords3d(loc_2d[0], loc_2d[1], height)
                d.append(tx_coords.get_distance_to(rx_coords))
                req_power, gain =\
                    get_req_power_and_gain(tx_coords.get_distance_to(rx_coords), beamwaist_radius=BEAMWAIST_RADII[i],
                                           tx_coords=tx_coords, rx_coords=rx_coords)
                uav_req_power.append(req_power)
                uav_gains.append(gain)
        res.append([uav_req_power, uav_gains])
        distances.append(d)

    min_gains = []
    for _res in res:
        min_gains.append(np.min(_res[1]))


    required_power = 1800

    required_tx = []
    for _gain in min_gains:
        required_tx.append(required_power * (1 + POWER_SPLIT_RATIO)/ _gain)
