"""

"""
from logging import getLogger

import numpy as np
import matplotlib.pyplot as plt

logger = getLogger(__name__)


class DetectorUnitNetwork:
    """
    classdocs
    """

    def __init__(self, name="NotDefined"):
        """
        Constructor
        """
        self.name = name

    def init_pos_id(self, du_pos, du_id):
        self.du_pos = du_pos
        self.du_id = du_id
        assert isinstance(self.du_pos, np.ndarray)
        assert isinstance(self.du_id, np.ndarray)
        assert du_pos.shape[0] == du_id.shape[0]
        assert du_pos.shape[1] == 3

    def get_sub_network(self, l_id):
        sub_net = DetectorUnitNetwork("sub-network of " + self.name)
        sub_net.init_pos_id(self.du_pos[l_id], self.du_id[l_id])
        return sub_net

    def get_pos_id(self, l_id):
        pass

    def get_surface(self):
        pass

    def get_max_dist_du(self, l_id):
        pass

    def plot_du_pos(self):
        plt.figure()
        plt.title(f"{self.name}\nDU network")
        for du_idx in range(self.du_pos.shape[0]):
            plt.plot(self.du_pos[du_idx, 0], self.du_pos[du_idx, 1], "*")
        plt.grid()
        plt.ylabel("[m]")
        plt.xlabel("[m]")

    def plot_trace_vmax_network(self):
        pass
