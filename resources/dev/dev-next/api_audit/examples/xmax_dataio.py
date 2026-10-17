import os

from grand.dataio.root_files import get_simu_parameters

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"
EFIELD = "efield_1618-13790_L0_0000.root"

for label, folder in (("Xmax above sea level (the sample)", SAMPLE), ("Xmax above the ground", "above_ground/")):
    simu = get_simu_parameters(folder + EFIELD)
    print("%-34s xmax_pos_shc z %8.1f -> FIX_xmax_pos z %8.1f" % (label, simu["xmax_pos_shc"][2], simu["FIX_xmax_pos"][2]))
