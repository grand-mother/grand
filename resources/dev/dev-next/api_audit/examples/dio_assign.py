import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


from grand.dataio import TRun, TShower
from grand.dataio.descriptors import StdVectorList


def assign(tree, field, value):
    setattr(tree, field, value)
    stored = getattr(tree, field)
    return stored if isinstance(stored, (int, float, str)) else list(stored) if hasattr(stored, "__iter__") else stored


def set_item(vector, index, value):
    vector[index] = value
    return list(vector)


attempt("TRun().my_note = 'x'            ", lambda: assign(TRun(), "my_note", "x"))
attempt("TRun().run_number = 1.5         ", lambda: assign(TRun(), "run_number", 1.5))
attempt("TRun().run_number = '7'         ", lambda: assign(TRun(), "run_number", "7"))
attempt("TRun().origin_geoid = 5.0       ", lambda: assign(TRun(), "origin_geoid", 5.0))
attempt("TRun().du_xyz = [[1.0, 2.0]]    ", lambda: [list(row) for row in assign(TRun(), "du_xyz", [[1.0, 2.0]])])
attempt("TShower().energy_primary = 1e39 ", lambda: assign(TShower(), "energy_primary", 1e39))
attempt("StdVectorList('int', 3)         ", lambda: list(StdVectorList("int", 3)))
attempt("v = [1, 2, 3]; v[3] = 4         ", lambda: set_item(StdVectorList("int", [1, 2, 3]), 3, 4))
