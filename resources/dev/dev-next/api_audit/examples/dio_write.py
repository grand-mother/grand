import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


import ROOT
from grand.dataio import TRun, TShower


def shower(event_number):
    tree = TShower()
    tree.run_number, tree.event_number = 1, event_number
    tree.fill()
    return tree


def contents(name):
    f = ROOT.TFile(name)
    keys = sorted("%s;%d (%d entries)" % (k.GetName(), k.GetCycle(), f.Get(k.GetName()).GetEntries())
                  for k in f.GetListOfKeys())
    f.Close()
    return keys


shower(1).write("a.root")
attempt("1. a second tshower written to a.root", lambda: shower(2).write("a.root"))
print("   a.root holds", contents("a.root"))

shower(1).write("b.root")
run = TRun()
run.run_number = 1
run.fill()
attempt("2. a trun written to b.root with overwrite=True", lambda: run.write("b.root", overwrite=True))
print("   b.root holds", contents("b.root"))
