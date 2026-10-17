import os

from grand.dataio import TADC

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"

reader = TADC(SAMPLE + "adc_1618-13790_L1_0000.root")   # reads the sample
result = TADC()                                          # a new tree, in memory, for results
print("the new tree is in:", os.path.basename(result._tree.GetDirectory().GetName()))
del reader                                               # done reading
result.du_id = [1]
result.fill()
print("filled:", result.get_number_of_entries(), "entry")
