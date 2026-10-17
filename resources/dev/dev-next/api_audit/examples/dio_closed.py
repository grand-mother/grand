import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


from grand.dataio import TEfield, TShower
from grand.dataio.data_handling import DataFile

tefield = TEfield(SAMPLE + "efield_1618-13790_L0_0000.root")
tefield.stop_using()
attempt("1. after tefield.stop_using(): tefield.get_number_of_entries()", lambda: tefield.get_number_of_entries())

shower = TShower()
shower.run_number, shower.event_number = 1, 1
shower.fill()
attempt("2. stop_using() on a tree filled but never written", lambda: shower.stop_using())

data = DataFile(SAMPLE + "efield_1618-13790_L0_0000.root")
tree = data.tefield
data.close()
attempt("3. after data.close(): data.tefield.get_number_of_entries()", lambda: tree.get_number_of_entries())
