import os.path as osp
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

TPL_FMT_LOGGER = "%(asctime)s %(levelname)5s [%(name)s %(lineno)d] %(message)s"

DICT_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def check_logger_level(str_level):
    """Check the validity of the logger level specified in simu_config
    and choose the appropriate one.
    """
    try:
        return DICT_LOG_LEVELS[str_level]
    except KeyError:
        print(f"keyword '{str_level}' isn't in {DICT_LOG_LEVELS.keys()}, use warning level by default.")
        time.sleep(1)
        return DICT_LOG_LEVELS["warning"]


class MyFormatter(logging.Formatter):
    """Formatter without date and with millisecond by default"""

    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        """!Define my specific format for log
        
        Doesn't used directly by user, so never call in grand lib
         
        @param record: logger internal param 
        @param datefmt:logger internal param 
        """
     
        my_convert = self.converter(record.created)
        if datefmt:
            str_date = my_convert.strftime(datefmt)
        else:
            str_time = my_convert.strftime("%H:%M:%S")
            str_date = "%s.%03d" % (str_time, record.msecs)
        return str_date


class HandlerForLoggerGrand(object):
    """Logger with handler, formatter for GRAND"""

    def __init__(self, log_level="info", log_dest_path=None, stream=True, root="grand"):
        """!Create a logger with handler for grand
        
        @param log_level: standard python logger level : "debug", "info", ...               
        @param log_dest_path: create a log file with path and name log_dest_path
        @param stream: enable standard output
        @param root: define a root logger
        """
        self.logger = logging.getLogger(root)
        self.logger.setLevel(check_logger_level(log_level))
        self.log_level = log_level
        formatter = MyFormatter(fmt=TPL_FMT_LOGGER)
        if log_dest_path is not None:
            fh = logging.FileHandler(log_dest_path, mode="w")
            fh.setLevel(check_logger_level(self.log_level))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.filehandler = fh
        if stream:
            ch = logging.StreamHandler()
            ch.setLevel(check_logger_level(self.log_level))
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.streamhandler = ch        
        self.t_start = datetime.now()
    
    def message_start(self):
        """!
        Add message start with date and time
        """
        self.t_start = datetime.now()
        logger.info(f"===========> Start at {get_now_string()} <===========")
        logger.info("")

    def message_end(self):
        """!
        Add message end with date, time and duration between start message         
        """
        logger.info("")
        logger.info(f"===========> End at {get_now_string()} <===========")
        logger.info("")
        logger.info(f"Duration (h:m:s): {datetime.now()-self.t_start}")


def get_logger_path(pfile):
    """!

    @param pfile: give __file__ where this is function is call
    @return: grand.xx.yy.zz of module that call this function
    """
    l_sep = osp.sep
    p_grand = pfile.find(l_sep + "grand" + l_sep)
    if p_grand is None:
        return None
    g_str = pfile[p_grand + 1 : -3].replace(l_sep, ".")
    print(g_str)
    return g_str


def get_now_string():
    """!
    Returns string with current date, time
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
