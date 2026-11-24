"""
TEMP mem (M0): memory diagnostics for the event loop of process_file().
Delete this file (and the "# TEMP mem (M0)" lines of process_file.py) once the diagnosis is done.

Logs the CURRENT RSS (log_memory() reports ru_maxrss, the peak, which never decreases), the RSS
change accumulated by each pipeline stage over all events, and the size of the containers that
could grow with the number of events (grand_tree_list, open ROOT files, TRecons entries).

Environment switches (same code for every diagnostic run):
    PIPE_DIAG_SKIP=trecons,ewrite,shower  skip trecons.fill()+write(), e.write(), e.write_shower()
    PIPE_TRACEMALLOC=1                     trace Python+numpy allocations (slower, one run only)
"""
import collections
import logging
import os
import resource
import sys
import tracemalloc

import ROOT
from grand.dataio import grand_tree_list

logger = logging.getLogger("grand.process")

DIAG_SKIP = {s for s in os.environ.get("PIPE_DIAG_SKIP", "").split(",") if s}
_TRACEMALLOC = os.environ.get("PIPE_TRACEMALLOC") == "1"
_PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20


def rss_mb() -> float:
    """Current resident set size in MB."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE_MB


def peak_mb() -> float:
    """Peak resident set size in MB (same value as log_memory())."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class MemProbe:
    """Sums, for each pipeline stage, the RSS change since the previous stage, over all events.

    A stage that keeps memory accumulates a positive total; transient allocations cancel out.
    """

    def __init__(self, trecons, event_list):
        self.trecons = trecons
        self.el = event_list
        self.delta = {}
        self.count = {}
        self.event = {}
        self.n_written = 0
        self.snapshot = None
        if _TRACEMALLOC:
            tracemalloc.start()
            self.snapshot = tracemalloc.take_snapshot()
        self.last = rss_mb()

        d = trecons._tree.GetDirectory()
        f = trecons._tree.GetCurrentFile()
        logger.info(
            "MEM start root=%s python=%s trecons_dir=%s trecons_dir_class=%s trecons_dir_writable=%s "
            "trecons_file=%s gDirectory=%s",
            ROOT.gROOT.GetVersion(), sys.version.split()[0],
            d.GetName() if d else None, d.ClassName() if d else None, d.IsWritable() if d else None,
            f.GetName() if f else None, ROOT.gDirectory.GetName(),
        )
        if DIAG_SKIP:
            logger.warning("MEM diagnostic run: skipping %s", ",".join(sorted(DIAG_SKIP)))

    def mark(self, stage):
        now = rss_mb()
        self.delta[stage] = self.delta.get(stage, 0.0) + now - self.last
        self.count[stage] = self.count.get(stage, 0) + 1
        self.event[stage] = now
        self.last = now

    def start_event(self, i_evt, event_number):
        """Called first in each loop iteration: RSS before get_event, periodic report."""
        self.event = {}
        self.mark("between")
        if i_evt % 50 == 0:
            self.report(i_evt, event_number)

    def written(self):
        """Called after trecons.fill(): per-stage RSS of this event for a few written events."""
        self.n_written += 1
        if self.n_written <= 3 or self.n_written % 25 == 0:
            logger.info(
                "MEM written n=%d %s",
                self.n_written, " ".join(f"{k}={v:.1f}" for k, v in self.event.items()),
            )

    def report(self, i_evt, event_number, tag="loop"):
        e = self.el.event
        trees = collections.Counter(type(t).__name__ for t in grand_tree_list)
        files = collections.Counter(
            os.path.basename(f.GetName()).split("_")[0] for f in ROOT.gROOT.GetListOfFiles()
        )
        traced = ""
        if _TRACEMALLOC:
            traced = f" traced={tracemalloc.get_traced_memory()[0] / 2**20:.1f}"
        logger.info(
            "MEM sample i=%d evt=%s tag=%s written=%d rss=%.1f peak=%.1f pyblocks=%d gtl=%d files=%d "
            "trecons=%d trecons_list=%d e_trees=%s e_event_trees=%s%s | stages %s | gtl_by_type %s "
            "| files_by_prefix %s",
            i_evt, event_number, tag, self.n_written, rss_mb(), peak_mb(), sys.getallocatedblocks(),
            len(grand_tree_list), ROOT.gROOT.GetListOfFiles().GetSize(),
            self.trecons._tree.GetEntries(), len(self.trecons._entry_list),
            None if e._trees is None else len(e._trees),
            None if e._event_trees is None else len(e._event_trees),
            traced,
            " ".join(f"{k}={v:+.1f}/{self.count[k]}" for k, v in self.delta.items()),
            dict(trees), dict(files),
        )
        if _TRACEMALLOC and i_evt % 100 == 0:
            snapshot = tracemalloc.take_snapshot().filter_traces(
                [tracemalloc.Filter(False, tracemalloc.__file__)]
            )
            for stat in snapshot.compare_to(self.snapshot, "lineno")[:10]:
                logger.info("MEM traced_growth i=%d %s", i_evt, stat)
            self.snapshot = snapshot
