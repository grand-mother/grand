#! pylint: disable=line-too-long
r"""Logging for GRANDlib and for the scripts that use it.

Library modules only create a logger and write to it; they never configure
where the messages go.  That is left to the script that runs them, as the
`Python logging guide
<https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library>`_
recommends.  In a library module::

    from logging import getLogger

    logger = getLogger(__name__)

    def foo(var):
        logger.debug("call foo()")
        logger.info(f"var={var}")

A script chooses the output and the level with
:func:`create_output_for_logger`, and gets its own logger with
:func:`get_logger_for_script`, since ``__name__`` is ``"__main__"`` there.
:func:`string_begin_script`, :func:`string_end_script`, :func:`chrono_start`
and :func:`chrono_string_duration` format the usual start, end and timing
lines.

Examples
--------
A script that logs to the terminal and to ``log.txt`` at debug level::

    import grand.manage_log as mlg

    logger = mlg.get_logger_for_script(__file__)
    mlg.create_output_for_logger("debug", log_file="log.txt", log_stdout=True)

    logger.info(mlg.string_begin_script())
    logger.info(mlg.chrono_start())
    ...                                   # the work
    logger.info(mlg.chrono_string_duration())
    logger.info(mlg.string_end_script())

Each line of the log reads::

    11:28:09.621  INFO [grand.sim.efield2voltage 412] message

To reuse this module in another project, change ``NAME_PKG_GIT`` and
``NAME_ROOT_LIB``.
"""
# pylint: enable=line-too-long

import os.path as osp
import logging
from datetime import datetime
import time

# value to customize for each project
NAME_PKG_GIT = "grand"
NAME_ROOT_LIB = "grand"

# constant value to manage logger and its features
TPL_FMT_LOGGER = "%(asctime)s %(levelname)5s [%(name)s %(lineno)d] %(message)s"

DICT_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

START_BEGIN = datetime.now()
START_CHRONO = datetime.now()

SCRIPT_ROOT_LOGGER = ""

logger = logging.getLogger(__name__)

#############################
# Public functions of module
#############################


#: Log files create_output_for_logger() opened in this process
_log_files_written = set()


def create_output_for_logger(
    log_level="info", log_file=None, log_stdout=True, log_root=NAME_ROOT_LIB
):
    """Create a logger with handler for grand

    Parameters
    ----------
    log_level : str, optional
        Threshold: ``"debug"``, ``"info"``, ``"warning"``, ``"error"`` or ``"critical"``.
    log_file : str, optional
        File to write to; none by default.
    log_stdout : bool, optional
        Also write to standard output.
    log_root : str, optional
        Logger name to configure.
    """
    if isinstance(log_root, str):
        l_log_root = [log_root]
    else:
        # A copy: the caller's list was extended with SCRIPT_ROOT_LOGGER (#256)
        l_log_root = list(log_root)
    if SCRIPT_ROOT_LOGGER != "":
        l_log_root.append(SCRIPT_ROOT_LOGGER)
    ret_level = _check_logger_level(log_level)
    formatter = _MyFormatter(fmt=TPL_FMT_LOGGER)
    root = l_log_root[0]
    my_logger = logging.getLogger(root)
    my_logger.setLevel(ret_level)
    # Each call added handlers on top of the previous ones, so every message
    # was printed once per call, and truncated the log file (#256): the
    # handlers this function added before are replaced, and a file it already
    # wrote to in this process is appended to
    for name in l_log_root:
        target = logging.getLogger(name)
        for handler in list(target.handlers):
            if getattr(handler, "_grand_managed", False):
                target.removeHandler(handler)
    # first root logger NAME_ROOT_LIB define handler
    if log_file is not None:
        mode = "a" if osp.abspath(log_file) in _log_files_written else "w"
        _log_files_written.add(osp.abspath(log_file))
        f_hd = logging.FileHandler(log_file, mode=mode)
        f_hd._grand_managed = True
        f_hd.setLevel(ret_level)
        f_hd.setFormatter(formatter)
        my_logger.addHandler(f_hd)
    if log_stdout:
        s_hd = logging.StreamHandler()
        s_hd._grand_managed = True
        s_hd.setLevel(ret_level)
        s_hd.setFormatter(formatter)
        my_logger.addHandler(s_hd)
    # others root logger with handler already created
    for root in l_log_root[1:]:
        my_log = logging.getLogger(root)
        my_log.setLevel(ret_level)
        if log_file is not None:
            my_log.addHandler(f_hd)
        if log_stdout:
            my_log.addHandler(s_hd)
    mes_str = f"create handler for root logger: {l_log_root}"
    logger.info(mes_str)


def close_output_for_logger(log_root=NAME_ROOT_LIB):
    """close handler for test

    Parameters
    ----------
    log_root : str, optional
        Logger whose handlers to close.
    """
    my_logger = logging.getLogger(log_root)
    handlers = my_logger.handlers[:]
    for handler in handlers:
        handler.close()
        my_logger.removeHandler(handler)


def get_logger_for_script(pfile):
    """
    Return a logger with root logger is defined by the path of the file.

    @note
      Must be call before create_output_for_logger()

    Parameters
    ----------
    pfile : str
        Path of the calling script, used to name the logger.

    Returns
    -------
    logging.Logger
        A logger named after that script.
    """
    global SCRIPT_ROOT_LOGGER  # pylint: disable=global-statement
    str_logger = _get_logger_path(pfile)
    root_logger = str_logger.split(".")[0]
    if root_logger not in [NAME_PKG_GIT, NAME_ROOT_LIB]:
        SCRIPT_ROOT_LOGGER = root_logger
    return logging.getLogger(str_logger)


def string_begin_script():
    """
    Return string start message with date, time

    Returns
    -------
    str
        A banner marking the start of a script run.
    """
    global START_BEGIN  # pylint: disable=global-statement
    START_BEGIN = datetime.now()
    ret = f"\n===========> Begin at {_get_string_now()} <===========\n\n"
    return ret


def string_end_script():
    """
    Return string end message with date, time and duration

    Returns
    -------
    str
        A banner marking the end, with the elapsed time.
    """
    ret = f"\n\n===========> End at {_get_string_now()} <===========\n"
    ret += f"Duration (h:m:s): {datetime.now()-START_BEGIN}"
    return ret


def chrono_start():
    """
    Start chonometer

    Returns
    -------
    float
        The start time, to pass to :func:`chrono_string_duration`.
    """
    global START_CHRONO  # pylint: disable=global-statement
    START_CHRONO = datetime.now()
    return "-----> Chrono start"


def chrono_string_duration():
    """
    Return string with duration between call chrono_start()

    Returns
    -------
    str
        Elapsed time since :func:`chrono_start`, formatted.
    """
    return f"-----> Chrono duration (h:m:s): {datetime.now()-START_CHRONO}"


#########################################
# Internal functions of module
#########################################


def _check_logger_level(str_level):
    """Check the validity of the logger level specified

    Parameters
    ----------
    str_level : str
        Level name to validate.

    Returns
    -------
    int
        The matching :mod:`logging` level.

    Raises
    ------
    Exception
        If the name is not a known level.
    """
    try:
        return DICT_LOG_LEVELS[str_level]
    except KeyError:
        logger.error(
            f"keyword '{str_level}' isn't in {DICT_LOG_LEVELS.keys()}, "
            "use debug level by default."
        )
        time.sleep(1)
        return DICT_LOG_LEVELS["debug"]


def _get_string_now():
    """
    Returns string with current date, time

    Returns
    -------
    str
        The current time, formatted for a log line.
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_logger_path(pfile):
    """
    @return: NAME_PKG_GIT.xx.yy.zz of module that call this function

    Parameters
    ----------
    pfile : str
        Path of a script.

    Returns
    -------
    str
        Logger name derived from it.
    """
    l_sep = osp.sep
    r_str = l_sep + NAME_PKG_GIT + l_sep
    p_grand = pfile.find(r_str)
    if p_grand > 0:
        # -3 for size of ".py"
        g_str = pfile[p_grand + 1 : -3].replace(l_sep, ".")
    else:
        # out package git
        # -3 for size of ".py"
        logger.debug("out package git")
        if pfile[0] == l_sep:
            g_str = pfile[1:-3].replace(l_sep, ".")
        else:
            g_str = pfile[0:-3].replace(l_sep, ".")
    return g_str


class _MyFormatter(logging.Formatter):
    """Formatter without date and with millisecond by default"""

    converter = datetime.fromtimestamp  # type: ignore

    def formatTime(self, record, datefmt=None):
        """Define my specific time format for GRAND logger.

        @note
          This method is not used directly by the user.

        Parameters
        ----------
        record : logging.LogRecord
            Record being formatted.
        datefmt : str, optional
            Time format.

        Returns
        -------
        str
            The formatted timestamp.
        """
        my_convert = self.converter(record.created)
        if datefmt:
            str_date = my_convert.strftime(datefmt)
        else:
            str_time = my_convert.strftime("%H:%M:%S")
            str_date = f"{str_time}.{int(record.msecs):03d}"
        return str_date

    def format(self, record):
        r"""
        Override format function to manage multiline with \n

        @note
          This method is not used directly by the user.

        Parameters
        ----------
        record : logging.LogRecord
            Record to format.

        Returns
        -------
        str
            The formatted line.
        """
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg
