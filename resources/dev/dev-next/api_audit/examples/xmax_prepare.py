"""Copies the sample, storing Xmax above the ground as today's converter does."""
import os
import shutil

import ROOT

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"
NAME = "shower_1618-13790_L0_0000.root"
shutil.copytree(SAMPLE, "above_ground")
os.remove("above_ground/" + NAME)

run_file = ROOT.TFile.Open(SAMPLE + "run_1_L0_0000.root")
run = run_file.Get("trun")
run.GetEntry(0)
ground = run.origin_geoid[2]

source = ROOT.TFile.Open(SAMPLE + NAME)
showers = source.Get("tshower")
target = ROOT.TFile(f"above_ground/{NAME}", "RECREATE")
copy = showers.CloneTree(0)
for i in range(showers.GetEntries()):
    showers.GetEntry(i)
    showers.xmax_pos_shc[2] -= ground   # the same vector the copy fills from
    copy.Fill()
copy.BuildIndex("run_number", "event_number")   # dev looks events up by this index
copy.Write()
target.Close()
print("ground altitude", ground, "m; wrote above_ground/" + NAME)
