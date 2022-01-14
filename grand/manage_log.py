"""!
@brief
  Define output logger (file/stdout) for given level of message and some tools to use 
  logger in script.

@note
  This module can be copied and used in other projects by modifying the following 2 variables:
       - NAME_PKG_GIT
       - NAME_ROOT_LIB


@section log_mod How used python logger in library module

The best practice is indicated python documentation  
<a href="https://docs.python.org/3.8/howto/logging.html#advanced-logging-tutorial">
python documentation</a> 
is simply:
 
 @code{.py}

from logging import getLogger

logger = getLogger(__name__)

def foo(var):
  logger.debug('call foo()')
  logger.info(f"var={var}")
  ...
 @endcode

@warning
  Use always f-string to include current value of variables in message, or create a 
  string message with ".format" before and give it to logger.

and that's all. Nothing in "__init__.py". Now in a script

@section log_script How to define logger in a script and outputs

In a script the value of "__name__" is "__main__" so a specific logger definition is 
provided by this module by the function @link get_logger_for_script 
get_logger_for_script() @endlink called with "__file__" value.

The function @link create_output_for_logger create_output_for_logger() @endlink alllows to define 
output file/stdout and level 
of message.

A couple of function  can be useful to:
  - define message at the beginning and the end of script @link string_begin_script 
    string_xxx_script() @endlink
  - easily have chronometer @link chrono_start chrono_xxx() @endlink

Example:

 @code{.py}
import grand.manage_log as mlg

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_file="log.txt", log_stdout=True)

# specific logger definition for script because __mane__ is "__main__"
logger = mlg.get_logger_for_script(__file__)
logger.info(mlg.string_begin_script())
...


logger.info(mlg.chrono_start())

# Xmax, Efield, and input frame are all in shower frame.
field.voltage = antenna.compute_voltage(shower.maximum, field.electric, frame=shower.frame)

logger.info(mlg.chrono_string_duration())
...


logger.info(mlg.string_end_script())
plt.show()
 @endcode

 
"""

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

logger = logging.getLogger(__name__)


#########################################
# Internal functions of module
#########################################


def _check_logger_level(str_level):
    """!
    @brief
      Check the validity of the logger level specified
    """
    try:
        return DICT_LOG_LEVELS[str_level]
    except KeyError:
        print(
            f"keyword '{str_level}' isn't in {DICT_LOG_LEVELS.keys()}, "
            "use warning level by default."
        )
        time.sleep(1)
        return DICT_LOG_LEVELS["warning"]


class _MyFormatter(logging.Formatter):
    """!Formatter without date and with millisecond by default"""

    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        """!Define my specific time format for GRAND logger.

        @note
          This method is not used directly by the user.

        @param record: internal param
        @param datefmt: internal param
        """
        my_convert = self.converter(record.created)
        if datefmt:
            str_date = my_convert.strftime(datefmt)
        else:
            str_time = my_convert.strftime("%H:%M:%S")
            str_date = f"%{str_time}.{record.msecs:.3f}"
        return str_date

    def format(self, record):
        """!
        Override format function to manage multiline with \n

        @note
          This method is not used directly by the user.

        @param record: internal param
        """
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg


def _get_string_now():
    """!
    Returns string with current date, time
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_logger_path(pfile):
    """!
    @param pfile: give __file__ where this is function is call
    @return: NAME_PKG_GIT.xx.yy.zz of module that call this function
    """
    l_sep = osp.sep
    p_grand = pfile.find(l_sep + NAME_PKG_GIT + l_sep)
    if p_grand is None:
        return None
    # -3 for size of ".py"
    g_str = pfile[p_grand + 1 : -3].replace(l_sep, ".")
    print(g_str)
    return g_str


#############################
# Public functions of module
#############################


def create_output_for_logger(
    log_level="info", log_file=None, log_stdout=True, log_root=NAME_ROOT_LIB
):
    """!Create a logger with handler for grand

    @param log_level: standard python logger level define in DICT_LOG_LEVELS
    @param log_file: create a log file with path and name log_file
    @param log_stdout: enable standard output
    @param log_root: define a log_root logger
    """
    my_logger = logging.getLogger(log_root)
    my_logger.setLevel(_check_logger_level(log_level))
    formatter = _MyFormatter(fmt=TPL_FMT_LOGGER)
    if log_file is not None:
        f_hd = logging.FileHandler(log_file, mode="w")
        f_hd.setLevel(_check_logger_level(log_level))
        f_hd.setFormatter(formatter)
        my_logger.addHandler(f_hd)
    if log_stdout:
        s_hd = logging.StreamHandler()
        s_hd.setLevel(_check_logger_level(log_level))
        s_hd.setFormatter(formatter)
        my_logger.addHandler(s_hd)


def get_logger_for_script(pfile):
    """!
    Return a logger with root logger is defined by the path of the file
    @param pfile: path of the file, so always call with __file__ value
    """
    return logging.getLogger(_get_logger_path(pfile))


def string_begin_script():
    """!
    Return string start message with date, time
    """
    global START_BEGIN  # pylint: disable=global-statement
    START_BEGIN = datetime.now()
    ret = f"\n===========> begin at {_get_string_now()} <===========\n\n"
    return ret


def string_end_script():
    """!
    Return string end message with date, time and duration
    """
    ret = f"\n\n===========> End at {_get_string_now()} <===========\n"
    ret += f"Duration (h:m:s): {datetime.now()-START_BEGIN}"
    return ret


def chrono_start():
    """!
    Start chonometer
    """
    global START_CHRONO  # pylint: disable=global-statement
    START_CHRONO = datetime.now()
    return "-----> Chrono start"


def chrono_string_duration():
    """!
    Return string with duration between call chrono_start()
    """
    return f"-----> Chrono Duration (h:m:s): {datetime.now()-START_CHRONO}"
