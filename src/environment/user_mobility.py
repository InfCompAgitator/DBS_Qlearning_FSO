from src.environment import graph_tools
from src.data_structures import Coords3d
from random import randint
import numpy as np
from src.environment import plotting
from src.environment import obstacles
from multiprocessing import Process, Manager
from time import sleep
from src.parameters import *
from itertools import count
from datetime import date, datetime, time, timedelta
from dateutil import tz

from shutil import  rmtree

import os
from tqdm import tqdm

import time as timee

class UserWalker:
    _ids = count(0)

    def __init__(self, user_id, starting_vertex=None, initial_coords=None):
        if starting_vertex:
            self.last_waypoint = starting_vertex.id
            self.current_coords = starting_vertex.coords.copy()
            self.current_coords.z = UE_HEIGHT
            self.next_pause_time = 0
        elif initial_coords is not None:
            self.current_coords = Coords3d(initial_coords[0], initial_coords[1], UE_HEIGHT)
        else:
            raise ValueError('No coordinates were provided!')

        self.id = next(self._ids) if user_id is None else user_id
        self.waypoints = []
        self.speed = np.random.uniform(USER_SPEED[0], USER_SPEED[1])
        self.remaining_pause_time = 0

    def set_waypoints(self, waypoints, last_waypoint_id):
        assert (waypoints[0] == self.current_coords)
        waypoints.pop(0)
        self.waypoints = waypoints
        self.last_waypoint = last_waypoint_id

    def update_location(self, delta_t):
        """Update location towards next waypoint or pause. Returns True when pause
         is done or last waypoint reached"""
        if not self.remaining_pause_time:
            if self.waypoints:
                if self.current_coords.update(self.waypoints[0], delta_t * self.speed):
                    self.current_coords.set(self.waypoints[0])
                    self.waypoints.pop(0)
                return False
            else:
                if self.next_pause_time:
                    self.remaining_pause_time = self.next_pause_time
                    self.next_pause_time = 0
                else:
                    self.remaining_pause_time = randint(PAUSE_INTERVAL[0], PAUSE_INTERVAL[1])
        else:
            self.remaining_pause_time -= delta_t
            if self.remaining_pause_time <= 0:
                self.remaining_pause_time = 0

        if not self.remaining_pause_time:
            return True
        return False


class ObstaclesMobilityModel:
    loaded = False
    seeks = []
    loaded_duration = 0
    def __init__(self, time_step=TIME_STEP, number_of_users=NUM_OF_USERS, graph=None):
        if not graph:
            # Obtains default Madrid graph
            self.graph, self.area_bounds = graph_tools.get_graph_from_segments()
        self.obstacles_objects = self.graph.obstacles_objects
        self.graph_vertices = self.graph.get_vertices()
        self.number_of_vertices = len(self.graph_vertices)
        self.users = []
        self.plotter_func = plotter_func
        self.current_time = 0
        self.plot_flag = False
        self.users_coords = None
        self.plot_sleep = 0
        self.plotter_process = None
        self.time_step = time_step
        self.auxiliary_updates = []
        self.base_stations_coords = None
        self.n_users = number_of_users
        # self.reset_users()
        self.max_time_of_day = None
        self.starting_solar_hour, self.starting_solar_minute = STARTING_HOUR, STARTING_MINUTE
        self.time_of_day = datetime(2022, STARTING_MONTH, STARTING_DAY, STARTING_HOUR,
                                    STARTING_MINUTE,
                                    0, tzinfo=tz.gettz(TIME_ZONE))
        self.set_max_time_of_day(MAX_HOUR_DAY, 59)
        self.seconds_buffer = 0

    def reset_users(self, speed_divisor):
        self.current_time = 0
        if not self.loaded:
            for user_id in range(self.n_users):
                start_vertex_id = randint(0, self.number_of_vertices - 1)
                end_vertex_id = randint(0, self.number_of_vertices - 1)
                path_waypoints = self.graph.get_path_from_to(start_vertex_id, end_vertex_id)
                if len(self.users) > user_id:
                    self.users[user_id].last_waypoint.start_vertex_id
                    self.users[user_id].current_coords.set(self.graph_vertices[start_vertex_id].x,
                                             self.graph_vertices[start_vertex_id].y, UE_HEIGHT)
                    _user.set_waypoints(path_waypoints.copy(), end_vertex_id)
                else:
                    _user = UserWalker(user_id, self.graph_vertices[start_vertex_id])
                    _user.set_waypoints(path_waypoints.copy(), end_vertex_id)
                    self.users.append(_user)
        else:
            self.load_model(save_directory_in=self.loaded, duration=self.loaded_duration, max_num_users=self.n_users,
                            speed_divisor=speed_divisor)

    def generate_plot(self, plot_sleep=TIME_SLEEP):
        """If called, an animation of users mobility will be shown"""
        self.plot_flag = True
        self.plot_sleep = plot_sleep
        manager = Manager()
        self.users_coords = manager.list()
        for _user in self.users:
            self.users_coords.append([_user.current_coords.x, _user.current_coords.y])
        obstacles_list = self.obstacles_objects.obstaclesList
        self.plotter_process = Process(target=self.plotter_func,
                                       args=(self.graph.paths_segments, obstacles_list, self.users_coords,
                                             self.base_stations_coords if self.base_stations_coords else None))
        self.plotter_process.start()

    def update_plot_users_coords(self):
        for idx, _user in enumerate(self.users):
            self.users_coords[idx] = [_user.current_coords.x, _user.current_coords.y]

    def update_plot(self):
        self.update_plot_users_coords()
        if self.plot_sleep:
            sleep(self.plot_sleep)

    def update_users_locations(self):
        for _user in self.users:
            # if _user.id == 0:
            #     bla = 5
            if _user.update_location(self.time_step):
                if not self.loaded:
                    end_vertex_id = randint(0, self.number_of_vertices - 1)
                    path_waypoints = self.graph.get_path_from_to(_user.last_waypoint,
                                                                 end_vertex_id)
                    _user.set_waypoints(path_waypoints.copy(), end_vertex_id)
                else:
                    with open(os.path.join(self.loaded, f"user_{_user.id}"), 'rb') as _file:
                        _file.seek(self.seeks[_user.id])
                        _wayps = np.load(_file, allow_pickle=True)
                        waypoints = [Coords3d.from_array(_arr) for _arr in _wayps]
                        next_pause_time = int(np.load(_file, allow_pickle=True))
                        self.seeks[_user.id] = _file.tell()
                    _user.set_waypoints(waypoints.copy(), 0)
                    _user.last_waypoint = waypoints[-1]
                    _user.next_pause_time = next_pause_time

    def generate_model(self, duration=None):
        while duration is None or self.current_time < duration:
            if not self.generate_model_step():
                return

    def generate_model_step(self, time_step=None, static_users=False):
        self.current_time += self.time_step
        if self.current_time > self.loaded_duration and self.loaded_duration>0:
            return False
        if not static_users:
            self.update_users_locations()
        for _update in self.auxiliary_updates:
            _update()
        if self.plot_flag:
            self.update_plot()
        # return [_user.current_coords for _user in self.users]

    def get_obstacles(self):
        return self.obstacles_objects.obstaclesList

    def add_base_stations_to_plot(self, base_stations_coords):
        self.base_stations_coords = base_stations_coords

    def set_max_time_of_day(self, max_hour, max_minutes):
        if max_hour > self.time_of_day.time().hour:

            self.max_time_of_day = datetime.combine(self.time_of_day.date(), time(hour=max_hour, minute=max_minutes),
                                                    tzinfo=self.time_of_day.tzinfo)
        else:
            self.max_time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=max_hour, minute=max_minutes),
                                                    tzinfo=self.time_of_day.tzinfo)

    def increment_time(self, minutes_to_add=None):
        self.seconds_buffer += self.time_step
        if minutes_to_add is None and self.seconds_buffer > 60:
            minutes_to_add = int(self.seconds_buffer / 60)
            self.seconds_buffer = self.seconds_buffer % 60

            self.time_of_day = self.time_of_day + timedelta(minutes=minutes_to_add)
            if self.time_of_day > self.max_time_of_day:
                self.seconds_buffer = 0
                self.time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=self.starting_solar_hour,
                                                         minute=self.starting_solar_minute),tzinfo=self.time_of_day.tzinfo)
                self.set_max_time_of_day(MAX_HOUR_DAY, 59)

    def save_model(self, folder_name='C:\\Users\\user\\PycharmProjects\\droneFsoCharging\\src\\mobility_model_saved',
                   save_name="default", duration=60*60):
        """Speed, waypoints, pause_time, waypoints, pause_time, ..."""
        self.reset_users()
        assert os.path.isdir(folder_name)
        save_directory = os.path.join(folder_name, save_name)
        if os.path.isdir(save_directory):
            rmtree(save_directory)
        os.mkdir(save_directory)
        pbar = tqdm(total=len(self.users))
        for idx, _user in enumerate(self.users):
            self.current_time = 0
            with open(os.path.join(save_directory,f"user_{_user.id}"), 'ab') as _file:
                while duration is None or self.current_time < duration:
                    if self.current_time == 0:
                        np.save(_file, np.array(np.random.uniform(USER_SPEED[0], USER_SPEED[1])))
                    np.save(_file, np.array(_user.waypoints))
                    np.save(_file, np.array(randint(PAUSE_INTERVAL[0], PAUSE_INTERVAL[1])))
                    if _user.waypoints:
                        _user.current_coords.set(_user.waypoints[-1])

                    path_waypoints= []
                    while len(path_waypoints) < 2:
                        end_vertex_id = randint(0, self.number_of_vertices - 1)
                        path_waypoints = self.graph.get_path_from_to(_user.last_waypoint,
                                                                     end_vertex_id)
                    # path_waypoints.pop(-1)
                    _user.waypoints = path_waypoints.copy()
                    _user.last_waypoint = end_vertex_id
                    self.current_time += self.time_step
            pbar.update(idx)
        pbar.close()

    def load_model(self, folder_name='C:\\Users\\user\\PycharmProjects\\droneFsoCharging\\src\\mobility_model_saved',
                   save_name="default", duration=60*60*2, max_num_users=None, save_directory_in=None, speed_divisor=USER_SPEED_DIVISOR):
        save_directory = os.path.join(folder_name, save_name) if not save_directory_in else save_directory_in
        self.seeks = []
        n_users = len(os.listdir(save_directory))
        self.n_users = min(n_users, max_num_users if max_num_users else n_users)
        for uid in range(self.n_users):
            with open(os.path.join(save_directory, f"user_{uid}"), 'rb') as _file:
                _file.seek(0)
                speed = float(np.load(_file, allow_pickle=True))/ speed_divisor
                wyps = np.load(_file, allow_pickle=True)
                next_pause_time = int(np.load(_file, allow_pickle=True))
                self.seeks.append(_file.tell())
                if wyps.size == 0:
                    wyps = [Coords3d.from_array(_arr) for _arr in np.load(_file, allow_pickle=True)]
                    waypoints = [wyps[0]]
                else:
                    waypoints = [Coords3d.from_array(_arr) for _arr in wyps]
            starting_coords = Coords3d.from_array(waypoints[0])
            if len(self.users) > uid:
                self.users[uid].current_coords.set(starting_coords)
                self.users[uid].next_pause_time = next_pause_time
                self.users[uid].set_waypoints(waypoints.copy(), 0)
                self.users[uid].remaining_pause_time = 0
            else:
                _user = UserWalker(uid, initial_coords=starting_coords)
                _user.speed = speed
                _user.next_pause_time = next_pause_time
                _user.set_waypoints(waypoints.copy(), 0)
                self.users.append(_user)

        self.loaded = save_directory
        self.loaded_duration= duration
        self.users = tuple(self.users)

def plotter_func(path_segments, obstacles_list, users_coords, base_stations_coords):
    plotter = plotting.Plotter()
    background_objects = []
    for segment in path_segments:
        xs, ys = zip(*segment)
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='g', width=0.5, style='dotted')

    background_objects = []
    for _building in obstacles_list:
        xs, ys = zip(*_building.vertices + [_building.vertices[0]])
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='k', width=1, style='dashed')

    plotter.set_users(users_coords)
    if base_stations_coords:
        plotter.set_base_stations(base_stations_coords)
    plotter.start_plotter()


if __name__ == '__main__':
    mobility_model = ObstaclesMobilityModel()
    time1 = timee.time()
    mobility_model.load_model(save_name='extended_4_madrids_500_users', max_num_users=20)
    mobility_model.generate_plot()
    # mobility_model.save_model(save_name='extended_4_madrids_500_users_slow', duration=60*60*4)

    mobility_model.generate_model(duration=60*60*4)
    # print(timee.time() - time1)
    #
    # for j in range(100):
    #     mobility_model.generate_model_step(TIME_STEP)