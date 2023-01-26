import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.data_structures import Obstacle
from src.math_tools import line_point_angle, get_mid_azimuth
from src.parameters import UAVS_HEIGHTS, SUN_SEARCH_STEP, SUN_SEARCH_COUNT, MBS_LOCATION, UAVS_LOCATIONS, EXTEND_TIMES_FOUR

obstacles_madrid_list = [[9, 423, 129, 543, 52.5],
                         [9, 285, 129, 405, 49],
                         [9, 147, 129, 267, 42],
                         [9, 9, 129, 129, 45.5],
                         [147, 423, 267, 543, 31.5],
                         [147, 147, 267, 267, 52.5],
                         [147, 9, 267, 129, 28],
                         [297, 423, 327, 543, 31.5],
                         [297, 285, 327, 405, 45.5],
                         [297, 147, 327, 267, 38.5],
                         [297, 9, 327, 129, 42],
                         [348, 423, 378, 543, 45.5],
                         [348, 285, 378, 405, 49],
                         [348, 147, 378, 267, 38.5],
                         [348, 9, 378, 129, 42]]


# obstacles_madrid_list = [[9, 423, 129, 543, 52.5+60],
#                        [9, 285, 129, 405, 49+60],
#                        [9, 147, 129, 267, 42+60],
#                        [9, 9, 129, 129, 45.5+60],
#                        [147, 423, 267, 543, 31.5+60],
#                        [147, 147, 267, 267, 52.5+60],
#                        [147, 285, 267, 405, 28+60],
#                        [297, 423, 327, 543, 31.5+60],
#                        [297, 285, 327, 405, 45.5+60],
#                        [297, 147, 327, 267, 38.5+60],
#                        [297, 9, 327, 129, 42+60],
#                        [348, 423, 378, 543, 45.5+60],
#                        [348, 285, 378, 405, 49+60],
#                        [348, 147, 378, 267, 38.5+60],
#                        [348, 9, 378, 129, 42+60]]
#
# obstacles_madrid_list = [
#                        [9, 285, 129, 405, 49+60],
#                        [9, 147, 129, 267, 42+60],
#                        [9, 9, 129, 129, 45.5+60],
#
#                        [147, 147, 267, 267, 52.5+60],
#                        [147, 285, 267, 405, 28+60],
#
#                        [297, 285, 327, 405, 45.5+60],
#                        [297, 147, 327, 267, 38.5+60],
#                        [297, 9, 327, 129, 42+60],
#
#                        [348, 285, 378, 405, 49+60],
#                        [348, 147, 378, 267, 38.5+60],
#                        [348, 9, 378, 129, 42+60]]


class Obstacles(object):
    obstaclesList = []

    def __init__(self, obstacles_data_list, vertices_format='axes'):
        for obstacle_id, obstacle in enumerate(obstacles_data_list):
            vertices = []
            if vertices_format == 'coordinates':
                for idx in range(0, (len(obstacle) - 1), 2):
                    vertices.append((obstacle[idx], obstacle[idx + 1]))
            elif vertices_format == 'axes':
                vertices = [(obstacle[0], obstacle[1]), (obstacle[0], obstacle[3]),
                            (obstacle[2], obstacle[3]), (obstacle[2], obstacle[1])]
            self.obstaclesList.append(Obstacle(obstacle_id, obstacle[-1], vertices))

    def check_overlap(self, other_coords):
        for _obstacle in self.obstaclesList:
            if _obstacle.is_overlapping(other_coords.x, other_coords.y, other_coords.z):
                return True
        return False

    def get_total_vertices(self):
        total_vertices = []
        for obstacle in self.obstaclesList:
            total_vertices = total_vertices + obstacle.vertices
        return total_vertices

    def get_sunny_vertices(self, sun_azimuth, sun_elevation, drone_height=UAVS_HEIGHTS[0]):
        total_vertices = []
        if sun_elevation < 0:
            return []
        for obstacle in self.obstaclesList:
            for _vertex in obstacle.vertices:
                [adj_1, adj_2] = obstacle.get_adjacent_vertices(_vertex)
                offset_azimuth, max_azimuth, min_azimuth = get_mid_azimuth(_vertex, adj_1, adj_2)

                direction1 = line_point_angle(length=SUN_SEARCH_STEP * (SUN_SEARCH_COUNT - 1),
                                              point=[_vertex[0], _vertex[1]],
                                              angle_x=max_azimuth)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP * (SUN_SEARCH_COUNT - 1),
                                              point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth)
                x_2 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_2 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                direction1 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=max_azimuth + 180)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth + 180)

                x_1 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_1 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                x_ticks = np.roll(np.linspace(x_1, x_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT - 1)
                y_ticks = np.roll(np.linspace(y_1, y_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT - 1)

                idxes = np.array(np.meshgrid([np.arange(0, SUN_SEARCH_COUNT)],
                                             [np.arange(0, SUN_SEARCH_COUNT)])).T.reshape(-1, 2).T

                idxes = np.array(sorted(idxes.T.tolist(), key=lambda x: (x[0] + x[1] + ((x[0] * x[1]) == 0)) / 2))

                idxes = idxes[3:-1]
                # idxes = idxes[(idxes != 1).any(axis=1)]

                for idx_x, idx_y in idxes:
                    x_coord, y_coord = x_ticks[idx_x], y_ticks[idx_y]
                    not_valid = False
                    for blocking_obstacle in self.obstaclesList:
                        if blocking_obstacle.is_overlapping(x_coord, y_coord, drone_height):
                            not_valid = True
                            break
                        if blocking_obstacle.is_blocking(x_coord, y_coord, sun_azimuth, sun_elevation,
                                                         reference_height=drone_height):
                            not_valid = True
                            break
                    if not not_valid:
                        total_vertices.append([x_coord, y_coord])
                        break
        return total_vertices

    def get_total_edges(self):
        edges = []
        for obstacle in self.obstaclesList:
            obstacle_poly = obstacle.vertices + [obstacle.vertices[0]]
            for idx in range(len(obstacle.vertices)):
                edges.append([obstacle_poly[idx], obstacle_poly[idx + 1]])
        return edges

    def print_obstacles(self):
        for obstacle in self.obstaclesList:
            print(obstacle.id, ": ", obstacle.vertices, obstacle.height)

    def plot_obstacles(self, show_flag=False, fill_color=None):
        if not fill_color:
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                plt.plot(xs, ys, c='dimgray')
        else:
            rects = []
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                xs = set(xs)
                ys = set(ys)
                corner = min(xs), min(ys)
                height = max(ys) - min(ys)
                width = max(xs) - min(xs)
                rects.append(Rectangle(corner, width, height, color=fill_color))
            return rects

        if show_flag:
            print("SHOWING")
            plt.show()


def get_madrid_buildings():
    # for _bldg in obstacles_madrid_list:
    #     _bldg[0], _bldg[1], _bldg[2], _bldg[3] = _bldg[1], _bldg[0], _bldg[3], _bldg[2]
    extension = []
    if EXTEND_TIMES_FOUR:
        x_shift = 400 #Total dimensions for Madrid Grid is 387 m (east-west) and 552 m (south north).  The
                                    #   building height is uniformly distributed between 8 and 15 floors with 3.5 m per floor
        y_shift = 570
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0] + x_shift, _bldg[1], _bldg[2] + x_shift, _bldg[3], _bldg[4]])
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0], _bldg[1] + y_shift, _bldg[2], _bldg[3] + y_shift, _bldg[4]])
        for _bldg in obstacles_madrid_list:
            extension.append([_bldg[0] + x_shift, _bldg[1] + y_shift, _bldg[2] + x_shift, _bldg[3] + y_shift, _bldg[4]])
        new_obstacles = obstacles_madrid_list + extension
    return Obstacles(new_obstacles)


if __name__ == '__main__':
    _rects = get_madrid_buildings().plot_obstacles(False, fill_color='gray')
    plt.plot(MBS_LOCATION.x, MBS_LOCATION.y, c='blue', marker='s', label='MBS', linestyle='none')
    ax = plt.gca()
    for _rect in _rects:
        ax.add_patch(_rect)
    # xs = np.array([0, 15.0, 200.0])
    # ys = np.array([0.0, 390.0, 370.0])
    # plt.plot(xs, ys, c='red')
    # plt.plot([], [], c='red', label='FSO link')
    # plt.plot(250.0, 380.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    # # plt.plot(200.0, 370.0, c='green', marker='o', label='Hotspot center', linestyle='none')

    xs = [82.5, 165, 82.5, 165, 345.5, 263, 345.5, 263]
    ys = [113.5, 227, 479.5, 366, 113.5, 227, 479.5, 366]

    xs = np.array([[_1[0] for _1 in _2] for _2 in UAVS_LOCATIONS]).flatten()
    ys = np.array([[_1[1] for _1 in _2] for _2 in UAVS_LOCATIONS]).flatten()

    for x, y in zip(xs, ys):
        plt.plot(x, y, c='green', marker='o', linestyle='none')
        # circle = plt.Circle((x, y), 300, color='r')
        # ax = plt.gca()
        # ax.add_patch(circle)

    # plt.legend(loc="upper left")
    plt.show()
    # plt.savefig('plots/madrid_modified_plus_stations.eps', format='eps')
