from granddb.granddatalib import DataManager
import os
import argparse
import grand.manage_log as mlg
logger = mlg.get_logger_for_script(__name__)

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config",default="config.ini", help="Config file to use")
args = argParser.parse_args()
dm = DataManager(os.path.dirname(__file__)+"/"+args.config)

materialized_views = ['datamat']
for view in materialized_views:
    logger.info(f'refreshing {view}.')
    dm.database().execute_sql(str('refresh materialized view '+view))