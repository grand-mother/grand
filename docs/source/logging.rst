Logging
=======

GRANDlib reports what it is doing through Python's :mod:`logging`: which
files and models it loads, which event it is on, where it writes.  By
default, in your own code, these messages go nowhere.  This page shows how to
see them, in a script, a notebook or a batch job.

From the command line
---------------------

Every command-line tool takes a verbosity level:

.. code-block:: bash

    python scripts/convert_efield2voltage.py my_simulation --verbose debug
    python scripts/convert_voltage2adc.py my_simulation -v warning

The levels are ``debug``, ``info`` (the default), ``warning``, ``error`` and
``critical``.  ``sim2root.py`` and ``convert_efield2efield.py`` take
``--verbose`` too.  In batch jobs, ``warning`` keeps the logs short and still
shows every problem.  The ``RunSimPipe*.py`` drivers always log at ``debug``.

From Python
-----------

:func:`grand.manage_log.create_output_for_logger` sends GRANDlib's messages
to the terminal, a file or both:

.. code-block:: python

    import grand.manage_log as mlg

    mlg.create_output_for_logger("info", log_file="grand.log", log_stdout=True)

    from grand import Efield2Voltage
    Efield2Voltage("my_simulation", "voltage.root", seed=1).compute_voltage()

Each line names the time, the level and the module and line that wrote it:

.. code-block:: text

    09:28:12.001  INFO [grand.sim.detector.antenna_model 177] Loading GP300 antenna model produced by HFSS simulation package
    09:28:14.914  INFO [grand.sim.efield2voltage 467] Running on event_number: 13790, run_number: 1
    09:28:15.905  INFO [grand.sim.efield2voltage 1310] save result in voltage.root

Calling it again replaces the outputs rather than adding to them, so a
notebook cell can be re-run without doubling every line.  A second call in
the same process appends to the log file it already wrote.

To hear more from one part of GRANDlib only, open the output at ``debug``,
then set the levels of the loggers with the standard library:

.. code-block:: python

    import logging

    mlg.create_output_for_logger("debug")
    logging.getLogger("grand").setLevel(logging.WARNING)                     # the rest
    logging.getLogger("grand.sim.efield2voltage").setLevel(logging.DEBUG)    # this module

In your own script
------------------

To log your script's messages in the same format and file, get its logger
from :func:`~grand.manage_log.get_logger_for_script` before creating the
outputs:

.. code-block:: python

    import grand.manage_log as mlg

    logger = mlg.get_logger_for_script(__file__)
    mlg.create_output_for_logger("info", log_file="my_analysis.log")

    logger.info(mlg.string_begin_script())
    ...
    logger.info(mlg.string_end_script())          # with the elapsed time

Warnings are separate
---------------------

Problems with the input, such as a value outside its physical range or a
file skipped because of its name, are Python warnings
(:class:`~grand.basis.validate.GRANDlibWarning`), not log messages.  They
appear whatever the logging level.  :doc:`troubleshooting` shows how to turn
them into errors, which is useful in batch jobs, where a warning is easy to
miss.

Where next
----------

* :doc:`troubleshooting` for what the messages mean.
* :doc:`at_scale` for batch jobs.
