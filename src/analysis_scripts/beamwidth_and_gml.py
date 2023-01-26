import numpy as np

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    wavelength = WAVELENGTH
    lens_radius = RX_DIAMETER / 2

    wavelength / (np.pi * BEAMWAIST_RADIUS)

    tx_coords = MBS_LOCATION
    rx_coords = Coords3d(82.5, 113.5, MBS_HEIGHT)
    beamwaist_radius = 0.25e-3

    # xs = np.linspace(1, 500, 100)
    # gmls=[]
    # for x in xs:
    #     rx_coords.x = x
    #     distance = tx_coords.get_distance_to(rx_coords)
    #     atmospheric_loss = StatisticalModel.get_atmospheric_loss(distance)
    #     atm_turb_induced_fading = 1  # for completeness
    #     c_n = StatisticalModel.get_refraction_index(tx_coords, rx_coords)
    #     rho = StatisticalModel.get_coherence_length(c_n, wavelength, distance)
    #     beam_width = StatisticalModel.get_beamwidth(distance, rho)
    #     v1 = StatisticalModel.get_v1(beam_width)
    #     phi, theta = np.pi / 2, np.pi
    #     v2 = StatisticalModel.get_v2(v1, phi, theta)
    #     t1 = StatisticalModel.get_t1(v1)
    #     t2 = StatisticalModel.get_t2(v2, phi, theta)
    #     t = (t1 + t2) / 2
    #     a0 = StatisticalModel.get_max_fraction_a0(v1, v2)
    #     lambda_1, lambda_2 = StatisticalModel.get_ig_fluctuations_eigen_values(tx_coords, rx_coords, phi, theta,
    #                                                                            distance,
    #                                                                            lens_radius)
    #     gml = StatisticalModel.get_gml(a0, t, beam_width, misalignment=np.sqrt(lambda_1 + lambda_2))
    #     gmls.append(gml)
    #
    # plt.plot(xs, gmls, label='GML')
    # plt.legend(loc='lower left')
    # ax = plt.gca()
    # ax.set_xlabel('x [m]', fontsize=15, **csfont)
    # ax.set_ylabel('GML ', fontsize=15, **csfont)
    # # plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/charging_gain.eps', format='eps')
    # plt.show()
    #
    # gmls =[]
    # beamwidths = []
    # rx_coords = Coords3d(82.5, 479.5, 60)
    # distance = tx_coords.get_distance_to(rx_coords)
    # bw_rs = np.linspace(0.25e-5, 0.25, 1000)
    # for bw_r in bw_rs:
    #     c_n = StatisticalModel.get_refraction_index(tx_coords, rx_coords)
    #     rho = StatisticalModel.get_coherence_length(c_n, wavelength, distance)
    #     beam_width = StatisticalModel.get_beamwidth(distance, rho, beamwaist_radius=bw_r)
    #     beamwidths.append(beam_width)
    #     v1 = StatisticalModel.get_v1(beam_width)
    #     phi, theta = np.pi / 2, np.pi
    #     v2 = StatisticalModel.get_v2(v1, phi, theta)
    #     t1 = StatisticalModel.get_t1(v1)
    #     t2 = StatisticalModel.get_t2(v2, phi, theta)
    #     t = (t1 + t2) / 2
    #     a0 = StatisticalModel.get_max_fraction_a0(v1, v2)
    #     lambda_1, lambda_2 = StatisticalModel.get_ig_fluctuations_eigen_values(tx_coords, rx_coords, phi, theta,
    #                                                                            distance,
    #                                                                            lens_radius)
    #     gml = StatisticalModel.get_gml(a0, t, beam_width, misalignment=np.sqrt(lambda_1 + lambda_2))
    #     gmls.append(gml)
    #
    #
    #
    # plt.plot(bw_rs, gmls, label='GML')
    # plt.legend(loc='lower left')
    # ax = plt.gca()
    # ax.set_xlabel('Beamwaist radius [m]', fontsize=15, **csfont)
    # ax.set_ylabel('GML ', fontsize=15, **csfont)
    # # plt.savefig('C:/Users/user/PycharmProjects/droneFsoCharging/src/analysis_scripts/figures/charging_gain.eps', format='eps')
    # plt.show()

    rx_coords = Coords3d(263, 366, 100)
    # rx_coords = Coords3d(345.5, 479.5, 100)
    distance = tx_coords.get_distance_to(rx_coords)
    beamwaist_radius = 0.015
    beamwaist_radius1 = 0.00450695945945946
    beamwaist_radius2 = 0.01576810810810811
    beamwaist_radius3 = 0.004757207207207207
    beamwaist_radius4 = 0.01576810810810811
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
    # phi, _ = StatisticalModel.get_orientation_angles(tx_coords,
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
    capacity / (20 * 1024 * 1024 * 40)

    required_capacity = 20 * 1024 * 1024 * 40/bandwidth  # 20Mb per symbol for 40 users

    required_power = 2**(2*required_capacity+ 4 * (lambda_1 + lambda_2) /(t * beam_width**2 * np.log(2)))/a0 * 2 * np.pi * noise_power / (np.exp(1) * responsivity**2 * atmospheric_loss**2)
    2 ** (2 * required_capacity + 4 / (t * beam_width ** 2 * np.log(2)) ) / a0 * 2 * np.pi * noise_power / (
                np.exp(1) * responsivity ** 2 * atmospheric_loss ** 2)
