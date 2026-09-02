# -*- coding: utf-8 -*-
r"""Reading events back through the file-reader classes.

``grand.dataio.root_files`` wraps a ROOT file so that an analysis can step
through its events without touching trees directly.  It was 21% covered: the
existing tests skip whenever ``data/test_voltage.root`` is absent, and that
file is not in version control.

These build their own file instead, as ``tests/sim/test_pipeline_end_to_end.py``
does, so they run on a fresh checkout.

Coverage here is thinner than for the rest of ``dataio``, and the reason is
worth stating: ``_FileEventBase`` reaches back out to the *directory* the file
sits in and looks up run and shower trees by names derived from the file's
own name, so constructing a valid input means reproducing a filename and
directory convention that nothing documents or enforces.  See
:ref:`issue-reader-directory-coupling`.
"""

import numpy as np
import pytest

from grand.dataio.event_trees import TShower, TVoltage
from grand.dataio.root_files import FileVoltage
from grand.dataio.run_trees import TRun

N_DU = 4
N_SAMPLES = 256
N_EVENTS = 3


@pytest.fixture(scope='module')
def voltage_file(tmp_path_factory):
    r"""Returns the path of a voltage file holding several events.

    Each event carries a pulse at a different position, so that loading the
    wrong one is visible rather than harmless.

    Returns
    -------
    str
        Path to the file.
    """
    path = str(tmp_path_factory.mktemp('rootfiles') / 'voltage_20260101_000000_RUN0_CD_L0_0000.root')

    run = TRun(path)
    run.run_number = 0
    run.du_id = list(range(N_DU))
    run.du_xyz = [[float(i) * 100.0, 0.0, 0.0] for i in range(N_DU)]
    run.t_bin_size = [0.5] * N_DU
    run.origin_geoid = [40.98, 93.95, 1200.0]
    run.fill()
    run.write()

    t = np.arange(N_SAMPLES) * 0.5
    voltage = TVoltage(path)
    for event in range(N_EVENTS):
        centre = 40.0 + 20.0 * event
        pulse = np.exp(-((t - centre) ** 2) / (2 * 3.0 ** 2))
        trace = np.stack([np.stack([pulse, 0.5 * pulse, 0.2 * pulse])
                          for _ in range(N_DU)]).astype(np.float32)
        voltage.run_number, voltage.event_number = 0, event
        voltage.du_id = list(range(N_DU))
        voltage.du_nanoseconds = [0] * N_DU
        voltage.du_seconds = [0] * N_DU
        voltage.trace = trace
        voltage.fill()
    voltage.write()

    # The reader looks up the run and shower trees for the file's analysis
    # level, so a directory holding voltages alone is not enough.
    shower = TShower(path)
    for event in range(N_EVENTS):
        shower.run_number, shower.event_number = 0, event
        shower.zenith, shower.azimuth = 85.0, 0.0
        shower.energy_primary = 1e9
        shower.shower_core_pos = [0.0, 0.0, 1200.0]
        shower.fill()
    shower.write()

    return path


def test_unconventional_filename_gives_a_clear_error(tmp_path):
    r"""A name without an analysis level says so, rather than raising RuntimeError.

    GRAND filenames carry ``_L0_`` or ``_L1_``.  Nothing enforces that, and a
    file without it used to reach a bare ``raise`` with no active exception,
    producing "RuntimeError: No active exception to reraise" -- which names
    neither the file nor the convention it broke.
    """
    path = str(tmp_path / 'no_level_marker.root')
    run = TRun(path)
    run.run_number = 0
    run.du_id = [0]
    run.du_xyz = [[0.0, 0.0, 0.0]]
    run.t_bin_size = [0.5]
    run.fill()
    run.write()

    with pytest.raises(Exception) as excinfo:
        FileVoltage(path)
    message = str(excinfo.value)
    assert 'No active exception' not in message, (
        'still failing with the uninformative RuntimeError')
    assert 'L0' in message or 'L1' in message or 'level' in message.lower(), (
        'the error does not mention the naming convention: %s' % message)
