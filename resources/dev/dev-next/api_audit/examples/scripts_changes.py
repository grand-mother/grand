import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from grand.dataio import TADC

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
SCRIPTS = os.path.join(os.environ["PYTHONPATH"], "scripts")   # the scripts of the version under test
ENV = dict(os.environ, MPLBACKEND="Agg")


def run(*args):
    done = subprocess.run([sys.executable, os.path.join(SCRIPTS, args[0])] + list(args[1:]),
                          capture_output=True, text=True, env=ENV)
    last = (done.stderr.strip().splitlines() or done.stdout.strip().splitlines() or [""])[-1]
    return done.returncode, last[:110]


os.mkdir("input")                                    # a copy: the scripts write next to their input
for name in glob.glob(SAMPLE + "/*.root"):
    shutil.copy(name, "input")

code, _ = run("convert_voltage2adc.py", "input", "-o", "out.root")
where = "input/out.root" if os.path.exists("input/out.root") else "out.root"
adc = TADC(where)
adc.get_entry(0)
print("1. convert_voltage2adc.py input -o out.root: exit", code, "| written to", where,
      "| traces", np.asarray(adc.trace_ch).shape)

with open("events.txt", "w") as f:
    f.write("%s,1,13790\n%s,1,99999\n" % (SAMPLE, SAMPLE))   # 99999 is not in the sample
os.mkdir("target")
print("2. extract_events.py with a missing event: exit, last line =", run("extract_events.py", "events.txt", "target"))
print("3. extract_rf_chain.py, first run: exit, last line =", run("extract_rf_chain.py"))
print("   extract_rf_chain.py, second run: exit, last line =", run("extract_rf_chain.py"))
print("4. plot_noise.py --lst 18.0: exit, last line =", run("plot_noise.py", "--lst", "18.0"))
