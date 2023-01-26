from src.parameters import *
from src.channel_model.fso_a2a import StatisticalModel
from src.types_constants import LinkType
from src.data_structures import Coords3d


class FsoTransceiver:
    def __init__(self, coords: Coords3d, endpoint: 'FsoTransceiver' = None, link_type=LinkType.A2A,
                 t_power=TX_POWER_FSO_DRONE, bs_id=None, is_backhaul=False):
        self.tx_power = t_power
        self.coords = coords
        self.link_type = link_type
        self.avg_gml = AvgGmlLoss[link_type]
        self.power_split_ratio = POWER_SPLIT_RATIO
        self.endpoint = endpoint
        self.bs_id = bs_id
        self.is_backhaul = is_backhaul
        self.link_capacity = 0
        self.received_charge_power = 0
        self.beamwaist_radius = BEAMWAIST_RADIUS

    def calculate_link_capacity_and_received_power(self):
        """ We assume that each link is rx, next endpoint is responsible for rx analysis"""
        if not self.is_backhaul:
            self.received_charge_power = 0
            self.link_capacity = self.endpoint.link_capacity
            return

        self.received_charge_power, self.link_capacity = StatisticalModel.get_charge_power_and_capacity(
            tx_coords=self.endpoint.coords, rx_coords=self.coords, transmit_power=self.endpoint.tx_power,
            beamwaist_radius=self.endpoint.beamwaist_radius)

    # def calculate_link_capacity_and_received_power1(self):
    #     """ We assume that each link is rx, next endpoint is responsible for rx analysis"""
    #     channel_gain = fsoMultiCasting.get_channel_gain(tx_coords=self.coords, rx_coords=self.endpoint.coords,
    #                                                     avg_gml=self.avg_gml)
    #     self.link_capacity = fsoMultiCasting.get_capacity(channel_gain=channel_gain, tx_power=self.endpoint.tx_power,
    #                                                  power_split_ratio=self.power_split_ratio)
    #
    #     self.received_charge_power = FSO_ENERGY_HARVESTING_EFF * (
    #             1 - self.power_split_ratio) * channel_gain * self.endpoint.tx_power if self.is_backhaul else 0
    #
    #     return self.link_capacity, self.received_charge_power

    def set_endpoint(self, endpoint: 'FsoTransceiver'):
        self.endpoint = endpoint
