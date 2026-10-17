import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


import time
from datetime import datetime, timezone

os.environ["TZ"] = "Europe/Paris"   # a laptop in France, two hours ahead of UTC in summer
time.tzset()

import ROOT  # noqa: E402
from grand.dataio import TRun  # noqa: E402
from grand.dataio.data_tree import DataTree  # noqa: E402


def stored(name):
    """The metadata as a dictionary, read with the static helper of DataTree."""
    f = ROOT.TFile(name)
    meta = DataTree.get_metadata_as_dict(f.Get("trun"))
    f.Close()
    return meta


def raw(name, key):
    """The value as stored in the file's metadata, without GRANDlib."""
    f = ROOT.TFile(name)
    value = f.Get("trun").GetUserInfo().FindObject(key).GetVal()
    f.Close()
    return "%d, which is %s UTC" % (value, datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M"))


run = TRun()
attempt("1. a new tree: modification_software and its version", lambda: (run.modification_software, run.modification_software_version))
run.creation_datetime = datetime(2024, 7, 1, 12, 0)
attempt("2. creation_datetime = datetime(2024, 7, 1, 12, 0), read back", lambda: run.creation_datetime)
run.run_number = 1
run.fill()
run.write("run.root")
attempt("3. written to run.root, then read: creation_datetime", lambda: stored("run.root")["creation_datetime"])
attempt("4. the number stored in run.root for creation_datetime", lambda: raw("run.root", "creation_datetime"))
attempt("5. the same file: source_datetime, never set          ", lambda: stored("run.root")["source_datetime"])
