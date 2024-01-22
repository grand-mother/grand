import sys, os, getopt
import grand.manage_log as mlg
from granddatalib import DataManager


import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config",default="config.ini", help="Config file to use")
argParser.add_argument('file', type=str, help='File to register')
args = argParser.parse_args()


dm = DataManager(os.path.dirname(__file__)+"/"+args.config)

try:
    print(dm.register_file(args.file))
except Exception as e:
    logger.error(f'Error when importing {path}. Skipping.')
    logger.error(f'Error was {e}.')