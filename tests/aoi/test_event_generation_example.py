# -*- coding: utf-8 -*-
r"""#218: examples/aoi/event_generation.py runs twice, and writes all its events.

Its second run failed with NotUniqueEvent (it appended to the first run's
file), and ``Event.close_files()`` closed the shared file after writing the
first tree, so writing the second crashed the process.
"""

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "aoi" / "event_generation.py"


def test_the_example_runs_twice_and_writes_every_event(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for attempt in (1, 2):
        done = subprocess.run([sys.executable, str(EXAMPLE)], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, (attempt, done.stdout[-1000:], done.stderr[-2000:])
    from grand.dataio import TEfield, TShower, TVoltage
    for cls in (TVoltage, TEfield, TShower):
        with cls(str(tmp_path / "dummy_example_events.root")) as tree:
            assert tree.get_number_of_entries() == 10, cls.__name__
