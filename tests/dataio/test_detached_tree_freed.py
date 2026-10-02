"""A tree that a closing write() took out of its file is freed by stop_using() (#223).

``sim2root.py -ef N`` writes and closes a file set every N events; each set
used to leave its trees in memory, about 4 MB per set.
"""
import gc

import numpy as np
import psutil

import grand.dataio.data_tree as data_tree
from grand.dataio import TEfield


def _cycle(path):
    efield = TEfield(str(path))
    efield.run_number, efield.event_number = 1, 1
    efield.du_id = list(range(20))
    efield.trace = np.zeros((20, 3, 2000), np.float32)
    efield.fill()
    efield.write(force_close_file=True)
    address = data_tree.ROOT.addressof(efield._tree)
    assert address in data_tree._detached_by_write
    efield.stop_using()
    assert efield._tree is None and address not in data_tree._detached_by_write


def test_the_detached_tree_is_released(tmp_path):
    _cycle(tmp_path / "one.root")


def test_memory_per_file_set_is_bounded(tmp_path):
    process = psutil.Process()
    for i in range(5):                       # settle ROOT's own first allocations
        _cycle(tmp_path / ("warm%d.root" % i))
    gc.collect()
    before = process.memory_info().rss
    for i in range(30):
        _cycle(tmp_path / ("set%d.root" % i))
    gc.collect()
    per_set = (process.memory_info().rss - before) / 30 / 2**20
    # 2.7 MB per set before the fix, 0.9 after (ROOT's own per-file share)
    assert per_set < 1.8, "%.2f MB per file set" % per_set
