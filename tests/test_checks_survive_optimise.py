# -*- coding: utf-8 -*-
r"""User-facing checks that were ``assert`` statements (#259).

Under ``python -O`` an ``assert`` disappears, and otherwise it fails with an
empty ``AssertionError``.  Each case runs under ``-O`` and must raise a
``GRANDlib:`` error.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

CASES = {
    "signal padding": "from grand.basis.signal import get_fastest_size_fft as f; f(100, 2000.0, 0.5)",
    "signal interpolation": ("import numpy as np; from grand.basis.signal import interpol_at_new_x as f; "
                             "f(np.zeros(0), np.zeros(0), np.ones(3))"),
    "galaxy interpolation": ("import numpy as np; from grand.sim.noise.galaxy import interpol_at_new_x as f; "
                             "f(np.zeros(0), np.zeros(0), np.ones(3))"),
    "network shape": ("import numpy as np; from grand.basis.du_network import DetectorUnitNetwork as N; "
                      "N().init_pos_id(np.zeros((3, 2)))"),
    "network ids": ("import numpy as np; from grand.basis.du_network import DetectorUnitNetwork as N; "
                    "N().init_pos_id(np.zeros((3, 3)), [1, 2])"),
    "electric field": ("import numpy as np; from grand.basis.type_trace import ElectricField; "
                       "from grand import CartesianRepresentation as C; "
                       "ElectricField(np.arange(4.0), C(x=np.zeros(5), y=np.zeros(5), z=np.zeros(5)))"),
    "antenna frequencies": ("import numpy as np; from grand.sim.detector.process_ant import AntennaProcessing; "
                            "AntennaProcessing.set_out_freq_mhz(object.__new__(AntennaProcessing), np.arange(1.0, 5.0))"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_check_raises_under_optimise(name):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    done = subprocess.run([sys.executable, "-O", "-c", CASES[name]], env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode != 0, "%s passed silently under -O" % name
    assert "GRANDlib:" in done.stderr, done.stderr[-1500:]
