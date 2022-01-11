
import os.path as osp
import logging
from datetime import datetime
import time


TEMPLATE = '%(asctime)s %(levelname)5s [%(name)s %(lineno)d] %(message)s'

DICT_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def check_logger_level(str_level):
    """Check the validity of the logger level specified in simu_config
    and choose the appropriate one.
    """
    try:
        return DICT_LEVELS[str_level]
    except KeyError:
        print(f"keyword '{str_level}' isn't in {DICT_LEVELS.keys()}, use warning level by default.")
        time.sleep(1)
        return DICT_LEVELS['warning']

    
class MyFormatter(logging.Formatter):
    """Formatter without date and with millisecond by default
    """
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None): 
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)                        
        return s

 
class LoggerGrand(object):
    """Logger Handler for GRAND
    """

    def __init__(self, log_level='info', log_dest_path=None, stream=True, root='grand'):
        """**Constructor**

        :param cfg: configuration container from the configuration file
        :type cfg: ConfigParser
        :param log_dest_path: path where the log file will be created
        :type log_dest_path: str
        """
        self.logger = logging.getLogger(root)
        self.logger.setLevel(check_logger_level(log_level))
        self.log_level = log_level
        formatter = MyFormatter(fmt=TEMPLATE)
        if log_dest_path is not None:
            fh = logging.FileHandler(log_dest_path, mode='w')
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
        
        self.logger.info(f'===========> Start at {get_now_string()} <===========')
        self.logger.info('')

    def message_end(self):
        self.logger.info('')
        self.logger.info(f'===========> End at {get_now_string()}')
        self.logger.info('')
        self.logger.info(f'Duration (h:m:s): {datetime.now()-self.t_start}')


def get_logger_path(pfile):
    """    
    
    :param pfile: __file__ where this is function is call
    :type pfile: str
    :return: ecpi.xx.yy.zz of module that call this function
    """
    l_sep = osp.sep
    p_ecpi = pfile.find(l_sep + 'ecpi' + l_sep)
    if p_ecpi is None:
        return None
    g_str = pfile[p_ecpi + 1:-3].replace(l_sep, '.')
    return g_str


def get_now_string():
    """
    Returns string with current date, time
    """
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
