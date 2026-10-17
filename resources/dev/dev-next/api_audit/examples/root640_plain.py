import gc

import ROOT

tree = ROOT.TTree("t", "t")
tree.GetUserInfo().Add(ROOT.TNamed("comment", "x"))   # how GRANDlib adds a tree's metadata
gc.collect()
print("comment:", tree.GetUserInfo().FindObject("comment").GetTitle())
