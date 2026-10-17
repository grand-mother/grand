import sys
import ROOT

f = ROOT.TFile.Open(sys.argv[1])
for key in f.GetListOfKeys():
    tree = f.Get(key.GetName())
    print(key.GetName(), "entries:", tree.GetEntries(), "branches:", tree.GetListOfBranches().GetEntries())
