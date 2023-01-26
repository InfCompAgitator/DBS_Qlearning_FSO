from src.math_tools import db2lin, wh_to_joules, lin2db
from src.types_constants import LinkType
from src.data_structures import Coords3d
from numpy import exp
from numpy import log

#Obstacles
EXTEND_TIMES_FOUR = True

# Users Mobility Model
NUM_OF_USERS = 140
USER_SPEED = [0.5, 0.8]
PAUSE_INTERVAL = [0, 60]
TIME_STEP = 2.5  # Between subsequent users mobility model updates
TIME_SLEEP = 2  # Sleep between updates to allow plotting to keep up
BUILDINGS_AREA_MARGIN = 50
SIMULATION_TIME = 60 * 60 * 2
USER_SPEED_DIVISOR = 1

# SOLAR PANELS
PANEL_EFFICIENCY_FACTOR = 0.2
SOLAR_PANEL_AREA = 1
ABSORPTION_COEFFICIENT_CLOUD = 0.01

# SUN ENVIRONMENT
STARTING_DAY = 1
STARTING_MONTH = 7
STARTING_HOUR = 12  # 24 format
STARTING_MINUTE = 00
MAX_HOUR_DAY = 23
CLOUD_SPEED = 16 * 1
TIME_ZONE = 'Europe/Madrid'
SUN_SEARCH_STEP = 7  # m
SUN_SEARCH_COUNT = 5
MAX_SUN_SEARCH_STEPS = 10
BUILDING_EDGE_MARGIN = 1  # m across each axis
SHADOWED_EDGE_PENALTY = 100 / 3

# Channel model PLOS
PLOS_AVG_LOS_LOSS = 1
PLOS_AVG_NLOS_LOSS = 20
# PLOS_A_PARAM = 9.61
# PLOS_B_PARAM = 0.16
# PLOS_A_PARAM = 4.9 #Obtained using the method
# PLOS_B_PARAM = 0.4
PLOS_A_PARAM = 5.05 #Obtained using the method
PLOS_B_PARAM = 0.38

# Channel model RF
DRONE_TX_POWER_RF = 0.2  # W
USER_BANDWIDTH = 500e3 #*2  # Hz
DRONE_BANDWIDTH = 20e6  #+5e6 # Hz
MBS_BANDWIDTH = 20e6  # Hz
DEFAULT_CARRIER_FREQ_MBS = 2e9  # Hz
DEFAULT_CARRIER_FREQ_DRONE = 2e9 + MBS_BANDWIDTH  # Hz
NOISE_SPECTRAL_DENSITY = -174  # dBm/Hz
NOISE_POWER_RF = db2lin(NOISE_SPECTRAL_DENSITY - 30 + lin2db(USER_BANDWIDTH))  # dBm input -> linear in W
DEFAULT_SNR_THRESHOLD = db2lin(25)  # linear
MBS_TX_POWER_RF = 0.5  # W
SINR_THRESHOLD = db2lin(10)
ASSOCIATION_SCHEME = 'SINR'
USER_SINR_COVERAGE_HIST = 100

# Channel model FSO
RX_DIAMETER = 0.2  # m
DIVERGENCE_ANGLE = 0.06  # rads
RX_RESPONSIVITY = 0.5
AVG_GML = 3
WEATHER_COEFF = 4.3 * 10 ** -4  # /m
POWER_SPLIT_RATIO = 0.005
FSO_ENERGY_HARVESTING_EFF = 0.2
TX_POWER_FSO_DRONE = 0.2  # W
TX_POWER_FSO_MBS = 380  # W
BANDWIDTH_FSO = 1e9  # Hz
NOISE_VARIANCE_FSO = 0.8 * 1e-9
NOISE_POWER_FSO = 1e-6
EMPIRICAL_SNR_LOSSES = db2lin(15)  # Linear
BEAMWAIST_RADIUS = 0.25e-3
WAVELENGTH = 1550e-9
AvgGmlLoss = {LinkType.A2G: 3, LinkType.A2A: 3 / 1.5, LinkType.G2G: 5}  # TODO: Get refs


# # UAV (Energy, Speed, etc.)
# class UavParams:
UAV_MASS = 4  # kg
UAV_PROPELLER_RADIUS = 0.25  # [m]
NUMBER_OF_UAV_PROPELLERS = 4
AIR_DENSITY = 1.225 # kg/m2
GRAVITATION_ACCELERATION = 9.80665
PROFILE_DRAG_COEFFICIENT = 0.08
UAV_STARTING_ENERGY = wh_to_joules(222)  # wh
UAV_MAX_ENERGY = wh_to_joules(222)
UAV_MIN_ENERGY = 0
UAV_TRAVEL_SPEED = 13  # m/s
SKIP_ENERGY_UPDATE = False
SKIP_ENERGY_CHARGE = True

# GLOBECOM
# y_shift = 70
# x_shift = 30
# x_begin = 10
# y_begin = 20
# x_end = 428 - x_begin
# y_end = 593 - y_begin
# UAVS_LOCATIONS = [[[x_begin, y_end], [x_shift, y_end-y_shift]],
#                   [[x_begin, y_begin], [x_shift, y_shift]],
#                   [[x_end - x_shift, y_end-y_shift], [x_end, y_end]],
#                   [[x_end - x_shift, y_shift], [x_end, y_begin]]]

# # shift = 50
# # UAVS_LOCATIONS = [[[195, 275]], [[195, 525]], [[195, 25]], [[411, 400]], [[411, 131]],
# #                   [[-21, 400]], [[-21, 131]]]
# # new_locs = []
# # for _uav in UAVS_LOCATIONS:
# #     new_locs.append([[_uav[0][0] - shift, _uav[0][1]], [_uav[0][0] + shift, _uav[0][1]]])
# UAVS_LOCATIONS =[[[145, 275], [245, 275]], [[145, 525], [245, 525]],
#                  [[145, 25], [245, 25]], [[361, 400], [461, 400]],
#                  [[361, 131], [461, 131]], [[-71, 400], [29, 400]],
#                  [[-71, 131], [29, 131]]]
#
# UAVS_FREQS = [DEFAULT_CARRIER_FREQ_DRONE, DEFAULT_CARRIER_FREQ_DRONE + DRONE_BANDWIDTH, DEFAULT_CARRIER_FREQ_DRONE + DRONE_BANDWIDTH,
#               DEFAULT_CARRIER_FREQ_DRONE + 2*DRONE_BANDWIDTH, DEFAULT_CARRIER_FREQ_DRONE + 3*DRONE_BANDWIDTH,
#               DEFAULT_CARRIER_FREQ_DRONE + 3*DRONE_BANDWIDTH, DEFAULT_CARRIER_FREQ_DRONE + 2*DRONE_BANDWIDTH]

# shift = 50
# UAVS_LOCATIONS = [[[0 + shift, 0 + shift]],
#                   [[0 + shift, 552 - shift]],
#                   [[387 - shift, 552/2]],]

shift = 100
UAVS_LOCATIONS = [[[200 - shift, 300 - shift], [200 + shift, 300 + shift]],
                  [[600 + shift, 300 - shift], [600 - shift, 300 + shift]],
                  [[200 - shift, 900 + shift], [200 + shift, 900 - shift]],
                  [[600 - shift, 900 - shift], [600 + shift, 900 + shift]]]

# shift = 100
# UAVS_LOCATIONS = [[[200 - shift, 300 - shift], [200 + shift, 300 + shift], [200 - shift, 300 + shift], [200 + shift, 300 - shift]],
#                   [[600 + shift, 300 - shift], [600 - shift, 300 + shift], [600 + shift, 300 + shift], [600 - shift, 300 - shift]],
#                   [[200 - shift, 900 + shift], [200 + shift, 900 - shift], [200 - shift, 900 - shift], [200 + shift, 900 + shift]],
#                   [[600 - shift, 900 - shift], [600 + shift, 900 + shift], [600 - shift, 900 + shift], [600 + shift, 900 - shift]]]

# UAVS_LOCATIONS = [[[200, 300]], [[600, 300]], [[200, 900]], [[600, 900]]]


UAVS_HEIGHTS = [60]#, 80]#, 100]
# BEAMWAIST_RADII = [0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015]
BEAMWAIST_RADII = [0.01, 0.01, 0.02, 0.02]
NUM_MBS = 1
MBS_HEIGHT = 25  # m
DRONE_HEIGHT = UAVS_HEIGHTS[0]  # m
NUM_UAVS = len(UAVS_LOCATIONS)
UE_HEIGHT = 1.5  # To conform with channel models
# MBS_LOCATION = Coords3d(210, 0, MBS_HEIGHT)
MBS_LOCATION = Coords3d(400, 0, MBS_HEIGHT)
MAX_USERS_PER_DRONE = DRONE_BANDWIDTH/ USER_BANDWIDTH
# USER_MOBILITY_SAVE_NAME = 'users_200_truncated'
USER_MOBILITY_SAVE_NAME = 'extended_4_madrids_500_users'


class QLearningParams:
    ENERGY_LEVELS = [3340, 1800]  # Joules
    DISCOUNT_RATIO = 0.4
    TIME_STEP_Q = 25  # s
    CHECKPOINT_ID = 'FINAL_HIST100_STEP2_5'
    CHECKPOINTS_FILE = 'C:\\Users\\user\\PycharmProjects\\droneFsoCharging\\src\\machine_learning\\checkpoints'
    SAVE_ON = False #SIMULATION_TIME/TIME_STEP_Q - 1 -102#Save every N cycles
    LOAD_MODEL = True
    TESTING_FLAG = True  # Turns off exploration
    EXPLORATION_PROB = lambda x: 0.2
    # EXPLORATION_PROB = lambda x: 0.8 * exp(-0.00003 * x)
    LEARNING_RATE = lambda x: 0.4
    # LEARNING_RATE = lambda x: 1 / (x ** (2 / 8))
    RELIABILITY_LEVELS_QUANTIZER = log(3)
    FAIRNESS_LEVELS = 5
