Running at scale
================

This page is for processing many simulations, or a lot of measured data, on a
computing cluster.  The rule that keeps it simple: **one folder per job**.

.. note::

   Points marked **(to confirm)** depend on how the collaboration's
   computing is set up.  If you know the answer, please say so in a
   `GitHub issue <https://github.com/grand-mother/grand/issues>`_.

.. contents::
   :local:
   :depth: 1

One folder per job
------------------

Each command-line tool (:doc:`commands`) reads one simulation folder and
writes its output into that folder.  Give each job its own folder and jobs
never touch each other's files.

Two processes writing the same file are refused: a writer holds a lock on the
file until it closes it; the second writer stops with a message naming
it.  A resubmitted job whose first attempt is still running fails in the
same way instead of corrupting the output.  The lock needs a file system that
supports ``flock``; on one that does not, the check is skipped.

Give each job its own seed.  The same ``--seed`` in every job gives every
simulation the same noise.

What a job needs
----------------

Measured on the committed sample: two showers, 44 and 5 units, traces of
4 µs at 2 GHz, on one core:

=============================  ========  =================
Step                           Time      Peak memory
=============================  ========  =================
``convert_efield2voltage.py``  8 s       1.0 GB
``convert_voltage2adc.py``     5 s       0.65 GB
=============================  ========  =================

About 0.4 GB of each is ROOT and Python before any data are read.  Time
grows with the number of units and the trace length.  Run one folder of your
production on an interactive node first and size the jobs from it.

Each process uses one core for its own work.  When several jobs share a
node, set ``OMP_NUM_THREADS=1`` so the numerical libraries do not start a
thread per core in each of them.

A SLURM job array
-----------------

One array task per simulation folder, listed one per line in
``folders.txt``:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=grand-voltage
    #SBATCH --array=1-500%50          # 500 folders, at most 50 at a time
    #SBATCH --mem=2G
    #SBATCH --time=01:00:00
    #SBATCH --output=logs/%A_%a.log

    export OMP_NUM_THREADS=1
    source /path/to/grand/env/setup.sh

    folder=$(sed -n "${SLURM_ARRAY_TASK_ID}p" folders.txt)
    python /path/to/grand/scripts/convert_efield2voltage.py "$folder" \
        --seed "$SLURM_ARRAY_TASK_ID" --verbose warning
    python /path/to/grand/scripts/convert_voltage2adc.py "$folder" \
        --seed "$SLURM_ARRAY_TASK_ID" -v warning

At CC-IN2P3, GRAND jobs run on the ``htc`` partition with
``--account=grand`` and ``--licenses=sps`` to reach ``/sps``
**(to confirm for user accounts)**.  A GRANDlib environment is installed under
``/sps/grand/software/conda/`` **(to confirm which one users should
activate)**.

On a workstation
----------------

The same idea, with one process per core:

.. code-block:: python

    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    def voltage(job):
        seed, folder = job
        subprocess.run(["python", "scripts/convert_efield2voltage.py", str(folder),
                        "--seed", str(seed), "--verbose", "warning"], check=True)

    folders = sorted(Path("production").glob("sim_*"))
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(voltage, enumerate(folders, start=1)))

Use processes, not threads: ROOT's file handling is not safe to share between
threads.

Reading many files in one analysis
----------------------------------

Open one file at a time and release it before the next, as in the
:doc:`recipes` ("Many files without memory growth").  To treat a set of
files as one, give :class:`~grand.dataio.DataFile` a list of files or a
wildcard.

The production pipeline for measured data
-----------------------------------------

``scripts/pipeline/`` holds the Snakemake workflow that the production
account runs at CC-IN2P3 on each batch of files transferred from the sites.
Users do not run it; it is documented here so that you know where the
converted files come from.  For each raw file it:

1. registers the transfer in the GRAND database;
2. converts the binary to ROOT with ``gtot`` (``gtot_out.bash``), into
   ``<data_dir>/<site>/GrandRoot/<year>/<month>/``;
3. stores the result in iRODS and registers it in the database;
4. runs the monitoring on the 10-second files;
5. emails a report.

``pipeline_setup.env`` holds its paths and options and ``ccin2p3/`` the
Snakemake profile for SLURM.  ``cc_pipeline.bash`` submits one run.

Where next
----------

* :doc:`simulation_production` for what goes into the folders.
* :doc:`logging` to control what each job writes to its log.
