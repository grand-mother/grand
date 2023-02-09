from logging import getLogger
import os.path

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
        self.l_model_name = ["Light_GP300Antenna", "GP300Antenna"]
                    
    def set_antenna_model(self, type_file):
        if not type_file in self.l_model_name:
            logger.error('unknow model of Leff !')
            raise
            
        if type_file == "GP300Antenna":
            logger.info(f"Load model of antenna GP300")
            path_ant = grand_add_path_data_model("detector/GP300Antenna_EWarm_leff.npy")
            self.leff_ew = TabulatedAntennaModel.load(path_ant)
            path_ant = grand_add_path_data_model("detector/GP300Antenna_SNarm_leff.npy")
            self.leff_sn = TabulatedAntennaModel.load(path_ant)
            path_ant = grand_add_path_data_model("detector/GP300Antenna_Zarm_leff.npy")
            self.leff_z = TabulatedAntennaModel.load(path_ant)
            # convert Leff in cartesian
            self.leff_sn.table.compute_leff_cartesian()
            self.leff_ew.table.compute_leff_cartesian()
            self.leff_z.table.compute_leff_cartesian()
        elif type_file == "Light_GP300Antenna":
            path_ant = grand_add_path_data_model("detector/Light_GP300Antenna_EWarm_leff.npz")
            self.leff_ew = TabulatedAntennaModel.load(path_ant)
            path_ant = grand_add_path_data_model("detector/Light_GP300Antenna_SNarm_leff.npz")
            self.leff_sn = TabulatedAntennaModel.load(path_ant)
            path_ant = grand_add_path_data_model("detector/Light_GP300Antenna_Zarm_leff.npz")
            self.leff_z = TabulatedAntennaModel.load(path_ant)
        self.d_leff = {"sn": self.leff_sn, "ew": self.leff_ew, "z": self.leff_z}

    def plot_effective_length(self):
        pass
