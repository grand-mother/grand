"""The RF chain's per-frequency arrays can be released, and Efield2Voltage does (#284)."""
import pathlib
import shutil

import numpy as np
import pytest

from grand.sim.detector.rf_chain import RFChain

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def test_release_frees_the_frequency_arrays_and_keeps_the_inputs():
    freqs = np.arange(0, 251.0, 0.5)
    chain = RFChain()
    chain.compute_for_freqs(freqs)
    tf = np.array(chain.get_tf())
    chain.release_arrays()
    assert chain.Z_in is None and chain.total_ABCD_matrix is None
    assert chain.lna.s21 is None and chain.lna.ABCD_matrix is None
    assert chain.lna.sparams is not None and chain.lna.freqs_in is not None
    with pytest.raises(RuntimeError, match="RFChain.vout_f: the arrays were released"):
        chain.get_tf()
    chain.compute_for_freqs(freqs)                          # usable again
    np.testing.assert_array_equal(chain.get_tf(), tf)


@pytest.mark.skipif(not (ROOT / "data" / "detector").exists() or not SAMPLE.is_dir(),
                    reason="needs the data model and the RUN1 sample")
def test_efield2voltage_keeps_the_transfer_function_and_releases_the_chain(tmp_path):
    from grand import Efield2Voltage

    folder = tmp_path / "sample"
    shutil.copytree(SAMPLE, folder)
    signal = Efield2Voltage(str(folder), "out.root", output_directory=str(tmp_path), seed=1)
    signal.params["add_noise"] = False
    signal.compute_voltage(event_number=13790, run_number=1)
    assert signal.rf_chain.lna.s21 is None                  # released
    tf = signal._chain_tf["rf_chain"]
    chain = RFChain()
    chain.compute_for_freqs(signal.freqs_mhz)
    np.testing.assert_array_equal(tf, chain.get_tf())       # the same transfer function
