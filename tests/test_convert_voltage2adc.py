# -*- coding: utf-8 -*-
r"""Runs ``scripts/convert_voltage2adc.py`` on a directory, as the pipeline does.

Given no ``-o``, the script writes each ADC file beside its voltage file and
names it after it: ``voltage_1-1_L0_0000.root`` gives ``adc_1-1_L1_0000.root``.
The replacements used to be made on the whole path, so the first 'voltage' or
'L0' in a *directory's* name was replaced instead.  A directory such as
``/tmp/test_efield2voltage_x/`` above the run was enough: the script failed
with ``OSError: Failed to open file .../voltage_1-1_L1_0000.root``, or wrote
into whichever existing directory the changed path named.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np

from grand.dataio.data_handling import DataFile
from grand.dataio.event_trees import TVoltage
from grand.dataio.run_trees import TRun

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'convert_voltage2adc.py'

N_DU, N_SAMPLES = 2, 64


def _write_sim_dir(directory):
    r"""Writes a run file and a voltage file into `directory`, named as sim2root does.

    The traces are zeros: only where the output goes is under test.  A 2 ns
    bin is the ADC's own 500 MHz, so the script does not resample.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to create and fill.
    """
    directory.mkdir(parents=True)

    run = TRun(str(directory / 'run_1_L0_0000.root'))
    run.run_number, run.analysis_level = 1, 0
    run.du_id = list(range(N_DU))
    run.t_bin_size = [2.0] * N_DU
    run.fill()
    run.write()

    voltage = TVoltage(str(directory / 'voltage_1-1_L0_0000.root'))
    voltage.run_number, voltage.event_number = 1, 1
    voltage.analysis_level = 0
    voltage.du_id = list(range(N_DU))
    voltage.du_seconds = [0] * N_DU
    voltage.du_nanoseconds = [0] * N_DU
    voltage.trigger_position = [N_SAMPLES // 2] * N_DU
    voltage.trace = np.zeros((N_DU, 3, N_SAMPLES), dtype=np.float32)
    voltage.fill()
    voltage.write()


def test_the_adc_file_lands_beside_the_voltage_file(tmp_path):
    r"""The ADC file is named from the file name alone, and written beside it.

    The directory's name holds both 'voltage' and 'L0', so a replacement made
    anywhere else in the path moves the output.  Every file under `tmp_path`
    is listed afterwards, so an output written elsewhere fails the test as
    surely as a missing one.
    """
    sim = tmp_path / 'efield2voltage_L0' / 'sim_run'
    _write_sim_dir(sim)

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))
    completed = subprocess.run([sys.executable, str(SCRIPT), str(sim)],
                               env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, (
        'convert_voltage2adc.py failed:\n%s\n%s'
        % (completed.stdout[-2000:], completed.stderr[-2000:]))

    written = sorted(p for p in tmp_path.rglob('*') if p.is_file())
    assert written == [sim / 'adc_1-1_L1_0000.root', sim / 'run_1_L0_0000.root',
                       sim / 'voltage_1-1_L0_0000.root'], written
    assert DataFile(str(sim / 'adc_1-1_L1_0000.root')).tadc.get_number_of_entries() == 1
