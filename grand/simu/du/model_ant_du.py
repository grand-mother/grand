from logging import getLogger

from grand import grand_add_path_data_model
from grand.io.file_leff import TabulatedAntennaModel

logger = getLogger(__name__)


class AntennaModelGeneric:
    pass


class AntennaModelTrend:
    pass


class AntennaModelHorizon:
    pass


class AntennaModelGp300(AntennaModelGeneric):
    def __init__(self):
        logger.info(f"Load model of antenna GP300")
        path_ant = grand_add_path_data_model("detector/GP300Antenna_EWarm_leff.npy")
        self.leff_ew = TabulatedAntennaModel.load(path_ant)
        path_ant = grand_add_path_data_model("detector/GP300Antenna_SNarm_leff.npy")
        self.leff_sn = TabulatedAntennaModel.load(path_ant)
        path_ant = grand_add_path_data_model("detector/GP300Antenna_Zarm_leff.npy")
        self.leff_z = TabulatedAntennaModel.load(path_ant)
        self.d_leff = {"ew": self.leff_ew, "sn": self.leff_sn, "z": self.leff_z}

    def plot_effective_length(self):
        pass
