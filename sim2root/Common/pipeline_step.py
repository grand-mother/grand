r"""Runs one step of the simulation pipeline, with its output shown live.

Used by ``RunSimPipe.py``, ``RunSimPipeNoJitter.py`` and
``RunSimPipeADCNoise.py`` (issue #121).

Those scripts used to capture each step's output: ``communicate()`` held all
of it until the step ended, then printed stderr as one raw byte string; the
streaming variant in ``RunSimPipeADCNoise.py`` read it back through a pipe,
which was reported to freeze on 1000-event runs.  The step now writes
straight to the terminal (or to the job's log file): nothing is buffered in
between, so the output appears as it is produced, in order, and no pipe can
fill up.
"""

import subprocess
import sys


def run_step(cmd):
    r"""Runs a shell command, its output going directly where ours goes.

    Parameters
    ----------
    cmd : str
        The command line, run through the shell as before.

    Returns
    -------
    int
        The command's exit status.  A failed step is reported but does not
        stop the pipeline, as before: a step can fail at exit after writing
        good output (see issue #90).
    """
    print("about to run: " + cmd, flush=True)
    # Flush our own output first, so that it is not printed after the step's
    sys.stdout.flush()
    sys.stderr.flush()
    status = subprocess.run(cmd, cwd=".", shell=True).returncode
    if status != 0:
        print("WARNING: step exited with status %d: %s" % (status, cmd),
              file=sys.stderr, flush=True)
    return status
