import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


import shutil

import ROOT
from grand.dataio.data_handling import DataDirectory, DataFile

ADC = SAMPLE + "adc_1618-13790_L1_0000.root"
ROOT.TFile("empty.root", "RECREATE").Close()          # a ROOT file without GRAND trees
os.mkdir("renamed")                                   # the run file, with a lowercase level in its name
shutil.copy(SAMPLE + "run_1_L0_0000.root", "renamed/run_1_l0_0000.root")


def trees(data):
    return sorted(type(tree).__name__ for tree in data.tree_instances)


attempt("1. DataFile('file://' + ADC)                 ", lambda: trees(DataFile("file://" + ADC)))
attempt("2. DataFile('empty.root')                    ", lambda: trees(DataFile("empty.root")))
attempt("3. DataDirectory('no_such_folder')           ", lambda: type(DataDirectory("no_such_folder")).__name__)
attempt("4. DataDirectory('renamed').trun             ", lambda: type(DataDirectory("renamed").trun).__name__)
attempt("5. DataDirectory(SAMPLE, analysis_level=1.0) ", lambda: type(DataDirectory(SAMPLE, analysis_level=1.0).trun).__name__)
attempt("6. DataDirectory(SAMPLE).tadc_L1            ", lambda: DataDirectory(SAMPLE).tadc_L1)
