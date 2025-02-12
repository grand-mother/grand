import sys, os, getopt
import grand.manage_log as mlg
from granddb.datamanager import DataManager
import argparse
from icecream import ic

logger = mlg.get_logger_for_script(__name__)

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config", default="config.ini", help="Config file to use")
argParser.add_argument("-r", "--repository", default="", help="Repository")
argParser.add_argument('dirs', nargs='+', default=[], help='dirs to register')
args = argParser.parse_args()

# if config is given as absolute path, use it. If not then use path relative to script
if args.config[0] == '/':
    config_path = args.config
else:
    config_path = os.path.dirname(__file__)+"/"+args.config
ic(config_path)
dm = DataManager(config_path)
if args.repository == '':
    repo_name = None
else:
    repo_name = args.repository

for dir in args.dirs:
    try:
        logger.info(f'Register ${dir}')
        #from os import listdir
        #from os.path import isfile, join
        onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        ic(onlyfiles)
        for file in onlyfiles:
            lfile=os.path.join(dir,file)
            dataset=os.path.basename(os.path.normpath(dir))
            ic(dataset)
            dm.register_file(localfile=lfile, dataset=dataset, repository=repo_name, again=True)
    except Exception as e:
        logger.error(f'Error when importing {file}. Skipping.')
        logger.error(f'Error was {e}.')