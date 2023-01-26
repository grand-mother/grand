#! pylint: disable=line-too-long
"""
brief:
  Define output logger (file/stdout) for given level of message and some tools to use 
  logger in script.

note:
  This module can be copied and used in other projects by modifying the following 2 variables:
       - NAME_PKG_GIT
       - NAME_ROOT_LIB

Log_mod How used python logger in library module:

The best practice is indicated in
<a href="https://docs.python.org/3.8/howto/logging.html#advanced-logging-tutorial">
python documentation</a> 
In particular this note:

note:
  It is strongly advised that you do not add any handlers other than NullHandler to 
  your library’s loggers. This is because the configuration of handlers is the 
  prerogative of the application developer who uses your library. The application 
  developer knows their target audience and what handlers are most appropriate for 
  their application: if you add handlers ‘under the hood’, you might well interfere 
  with their ability to carry out unit tests and deliver logs which suit their 
  requirements.
  
and this one

note:
  A good convention to use when naming loggers is to use a module-level logger, 
  in each module which uses logging, named as follows:
 
 
from logging import getLogger

logger = getLogger(__name__)

def foo(var):
  logger.debug('call foo()')
  logger.info(f"var={var}")
  ...
 

:warning:
  Use always f-string to include current value of variables in message, or create a 
  string message with ".format" before and give it to logger.

and that's all. Nothing in "__init__.py". Now in a script

Log_script How to define logger in a script and outputs:

So the job of script is to define handler for logger, but script can also write log ans 
the value of "__name__" is "__main__" so a specific logger definition is 
provided by this module by the function @link get_logger_for_script 
get_logger_for_script() @endlink called with "__file__" value.

The function @link create_output_for_logger create_output_for_logger() @endlink alllows to define 
output file/stdout and level of message.

A couple of function  can be useful to:
  - define message at the beginning and the end of script @link string_begin_script 
    string_xxx_script() @endlink
  - easily have chronometer @link chrono_start chrono_xxx() @endlink

Example:


import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__"
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_file="log.txt", log_stdout=True)

logger.info(mlg.string_begin_script())
...

logger.info(mlg.chrono_start())

# Xmax, Efield, and input frame are all in shower frame.
field.voltage = antenna.compute_voltage(shower.maximum, field.electric, frame=shower.frame)

logger.info(mlg.chrono_string_duration())
...


logger.info(mlg.string_end_script())
plt.show()


Result log file

11:28:09.621  INFO [grand.examples.simulation.shower-event 27] 
11:28:09.621  INFO [grand.examples.simulation.shower-event 27] ===========> begin at 2022-01-17T11:28:09Z <===========
11:28:09.621  INFO [grand.examples.simulation.shower-event 27] 
11:28:09.621  INFO [grand.examples.simulation.shower-event 27] 
11:28:09.622  INFO [grand.simu.shower.gen_shower 102] Loading shower data from ../../tests/simulation/data/zhaires:/
11:28:09.625  INFO [grand.simu.shower.zhaires 104] ### zhaires.py: reading groundaltitude from. inp file.
11:28:10.030  INFO [grand.simu.shower.gen_shower 114] Loaded 176 field(s) from ../../tests/simulation/data/zhaires:/
11:28:10.030  INFO [grand.io.file_leff 73] Loading tabulated antenna model from /home/jcolley/.grand/HorizonAntenna_EWarm_leff_loaded.npy:/
11:28:10.077  INFO [grand.io.file_leff 80] Loaded 1841112 entries from /home/jcolley/.grand/HorizonAntenna_EWarm_leff_loaded.npy:/
11:28:10.183  INFO [grand.examples.simulation.shower-event 99] -----> Chrono start
11:28:10.183 DEBUG [grand.simu.du.process_ant 180] call compute_voltage()
11:28:10.244  INFO [grand.examples.simulation.shower-event 103] -----> Chrono Duration (h:m:s): 0:00:00.060869
11:28:10.396  INFO [grand.examples.simulation.shower-event 121] 
11:28:10.396  INFO [grand.examples.simulation.shower-event 121] 
11:28:10.396  INFO [grand.examples.simulation.shower-event 121] ===========> End at 2022-01-17T11:28:10Z <===========
11:28:10.396  INFO [grand.examples.simulation.shower-event 121] Duration (h:m:s): 0:00:00.775352
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


def create_output_for_logger(
    log_level="info", log_file=None, log_stdout=True, log_root=NAME_ROOT_LIB
):
    """Create a logger with handler for grand

    :param log_level: standard python logger level define in DICT_LOG_LEVELS
    :param log_file: create a log file with path and name log_file
    :param log_stdout: enable standard output
    :param log_root: define a log_root logger str or list of str
    """
    if isinstance(log_root, str):
        l_log_root = [log_root]
    else:
        l_log_root = log_root
    if SCRIPT_ROOT_LOGGER != "":
        l_log_root.append(SCRIPT_ROOT_LOGGER)
    ret_level = _check_logger_level(log_level)
    formatter = _MyFormatter(fmt=TPL_FMT_LOGGER)
    root = l_log_root[0]
    my_logger = logging.getLogger(root)
    my_logger.setLevel(ret_level)
    # first root logger NAME_ROOT_LIB define handler
    if log_file is not None:
        f_hd = logging.FileHandler(log_file, mode="w")
        f_hd.setLevel(ret_level)
        f_hd.setFormatter(formatter)
        my_logger.addHandler(f_hd)
    if log_stdout:
        s_hd = logging.StreamHandler()
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
    """close handler for test"""
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

    :param pfile: path of the file, so always call with __file__ value
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
    """
    global START_BEGIN  # pylint: disable=global-statement
    START_BEGIN = datetime.now()
    ret = f"\n===========> Begin at {_get_string_now()} <===========\n\n"
    return ret


def string_end_script():
    """
    Return string end message with date, time and duration
    """
    ret = f"\n\n===========> End at {_get_string_now()} <===========\n"
    ret += f"Duration (h:m:s): {datetime.now()-START_BEGIN}"
    return ret


def chrono_start():
    """
    Start chonometer
    """
    global START_CHRONO  # pylint: disable=global-statement
    START_CHRONO = datetime.now()
    return "-----> Chrono start"


def chrono_string_duration():
    """
    Return string with duration between call chrono_start()
    """
    return f"-----> Chrono duration (h:m:s): {datetime.now()-START_CHRONO}"


#########################################
# Internal functions of module
#########################################


def _check_logger_level(str_level):
    """Check the validity of the logger level specified"""
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
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_logger_path(pfile):
    """
    :param pfile: give __file__ where this is function is call
    @return: NAME_PKG_GIT.xx.yy.zz of module that call this function
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

        :param record: internal param
        :param datefmt: internal param
        """
        my_convert = self.converter(record.created)
        if datefmt:
            str_date = my_convert.strftime(datefmt)
        else:
            str_time = my_convert.strftime("%H:%M:%S")
            str_date = f"{str_time}.{int(record.msecs):03d}"
        return str_date

    def format(self, record):
        """
        Override format function to manage multiline with \n

        @note
          This method is not used directly by the user.

        :param record: internal param
        """
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg
