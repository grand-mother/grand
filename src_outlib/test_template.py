'''
TEMPLATE WITH LOGGER INITIALISED
'''

import matplotlib.pyplot as plt
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)


def test_foo():
    logger.info("do something")
    return False

if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #=============================================
    test_foo()
    #=============================================    
    logger.info(mlg.string_end_script())
    plt.show()
