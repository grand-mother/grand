from granddb.granddatalib import DataManager
import os
import argparse
from datetime import datetime

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config",default="config.ini", help="Config file to use")
argParser.add_argument("-s", "--status", help="Status of convertion", required=True)
argParser.add_argument("-i","--file",help="Bin file converted", required=True)
argParser.add_argument("-o","--root",help="Root file created", required=True)
argParser.add_argument("-l","--logfile",help="Logfile of convertion", required=True)

args = argParser.parse_args()

if args.config[0] == '/':
    config_path = args.config
else:
    config_path = os.path.dirname(__file__)+"/"+args.config

dm = DataManager(config_path)
logfile = os.path.normpath(args.logfile)
print(args.file)
print(args.status)
myfile = dm.database().sqlalchemysession.query(dm.database().tables()['rawfile']).filter_by(filename=args.file).first()
if not myfile:
    print("Error file not registered")
    exit(0)
else:
    id_raw_file = myfile.id_raw_file
    converted = {'id_raw_file': id_raw_file, 'date_convertion': datetime.now(), 'logfile': logfile, 'root_filename': args.root, 'retcode': args.status}
    container = dm.database().tables()['convertion'](**converted)
    dm.database().sqlalchemysession.add(container)
    dm.database().sqlalchemysession.commit()

dm.database().sqlalchemysession.close()