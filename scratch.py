#
# #PlosModel.get_optimal_height
# max_rs = []
# max_thetas = []
# for pl_max in range(100, 150, 10):
#     theta_plot = np.linspace(0, np.pi / 2, 10000)
#     r_plot = 10 ** (
#             (pl_max - A / (1 + env_a * np.exp(-env_b * (theta_plot - env_a))) - B) / 20) * np.cos(
#         theta_plot)
#     max_rs.append(max(r_plot))
#     max_thetas.append(theta_plot[np.argmin(r_plot)])
#
# env_a, env_b = env_b, env_a
# pls = []
# for pl_max in range(114, 121, 3):
#     hes = []
#     ras = []
#     errs = []
#     for he in range(1, 10000, 10):
#         min_error = 1000
#         for ra in range(1, 9000, 10):
#             # theta_2 = np.arctan(he/ra)
#             # error = abs(pl_max - (A / (1 + env_a * np.exp(-env_b * (theta_2 - a))) + 20 * np.log10(ra / np.cos(theta_2)) + B))
#             ue_coords = Coords3d(0, 0, 0)
#             uav_coords = Coords3d(ra / np.sqrt(2), ra / np.sqrt(2), he)
#             pl = PlosModel.get_path_loss(ue_coords, uav_coords)
#             error = abs(pl_max - lin2db(pl))
#             if error < min_error:
#                 min_error = error
#                 best_he = he
#                 best_ra = ra
#         if min_error < 1:
#             hes.append(best_he)
#             ras.append(best_ra)
#             errs.append(min_error)
#     pls.append([hes, ras, errs])
#
# theta_plot = np.linspace(0, np.pi, 10000)
# r_plot = 10 ** ((max_path_loss - A / (1 + env_a * np.exp(-env_b * (theta_plot - env_a))) - B) / 20) * np.cos(theta_plot)
#
# h = np.tan(theta_opt) * R
# theta_2 = np.arctan(h / R)
# max_pl = A / (1 + env_a * np.exp(-env_b * (theta_2 - a))) + 20 * np.log10(R / np.cos(theta_2)) + B
#
#
#
#
#
#
#
#
# timeit.timeit(stmt='np.nanmin([1,2,3,4,5,6,7])', setup='import numpy as np', number=1000000, globals=None)
#
# for bs_idx, bs_set in enumerate(self.stations_list):
#     for idx, bs in enumerate(bs_set):
#         path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
#         self.received_powers[bs_idx][idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)
#
# from time import time
# time1 = time()
# for i in range(1000000):
#     a = USER_SINR_COVERAGE_HIST
# print(time() - time1)


class Test:
    a = [1,2,3,4,5]
