"""rf_chain checks that survive ``python -O`` (#255)."""
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from grand.sim.detector import rf_chain

SOURCE = pathlib.Path(rf_chain.__file__)


def test_no_assert_statements_left():
    assert not [line for line in SOURCE.read_text().splitlines() if line.strip().startswith("assert ")]


def test_public_helpers_say_what_is_wrong():
    with pytest.raises(ValueError, match="interpol_at_new_x: 'a_x' is empty"):
        rf_chain.interpol_at_new_x(np.array([]), np.array([]), np.array([1.0]))
    with pytest.raises(ValueError, match=r"matmul: expects ABCD matrices of shape \(2, 2, n_freq\)"):
        rf_chain.matmul(np.zeros((3, 2, 4)), np.zeros((2, 2, 4)))


def test_an_empty_frequency_axis_is_refused_under_optimisation():
    code = ("import numpy as np\nfrom grand.sim.detector.rf_chain import LowNoiseAmplifier\n"
            "try:\n    LowNoiseAmplifier().compute_for_freqs(np.array([]))\n"
            "except ValueError as e:\n    print('refused', e)\n")
    done = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300)
    assert "refused" in done.stdout, done.stderr[-1500:]
