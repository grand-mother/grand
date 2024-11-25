import sys, os, getopt
import grand.manage_log as mlg
from granddatalib import DataManager
import argparse
logger = mlg.get_logger_for_script(__name__)

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config", default="config.ini", help="Config file to use")
argParser.add_argument("-r", "--repository", default="", help="Repository")
argParser.add_argument('files', nargs='+', default=[], help='Files to register')
args = argParser.parse_args()

# if config is given as absolute path, use it. If not then use path relative to script
if args.config[0] == '/':
    config_path = args.config
else:
    config_path = os.path.dirname(__file__)+"/"+args.config

dm = DataManager(config_path)
if args.repository == '':
    repo_name = None
else:
    repo_name = args.repository
for file in args.files:
    try:
        logger.info(f'Register ${file}')
        dm.register_file(file, None, repo_name)
    except Exception as e:
        logger.error(f'Error when importing {file}. Skipping.')
        logger.error(f'Error was {e}.')