from granddb.datamanager import DataManager
import os
import sqlite3
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config",default="config.ini", help="Config file to use")
argParser.add_argument("-d", "--database", help="Database file to use", required=True)
argParser.add_argument("-t", "--tag", default="*", help="Tag for the files to register")
args = argParser.parse_args()

if args.config[0] == '/':
    config_path = args.config
else:
    config_path = os.path.dirname(__file__)+"/"+args.config

dm = DataManager(config_path)

db = args.database
tag = args.tag

connection = sqlite3.connect(db)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()
cursor.execute("SELECT target as file, md5sum, transfer.success, transfer.date_transfer, transfer.comment, transfer.tag FROM gfiles, transfer WHERE gfiles.id = transfer.id AND transfer.tag = "+tag+";")
rows = cursor.fetchall()
connection.close()

for row in rows:
    trans = dict(row)
    rawfile = {'filename': os.path.basename(trans["file"]), 'md5': trans["md5sum"]}
    fname = os.path.basename(trans["file"])
    myobject = dm.database().sqlalchemysession.query(dm.database().tables()['rawfile']).filter_by(filename=fname).first()
    if not myobject:
        container = dm.database().tables()['rawfile'](**rawfile)
        dm.database().sqlalchemysession.add(container)
        dm.database().sqlalchemysession.flush()
        id_raw_file = container.id_raw_file
    else:
        id_raw_file = myobject.id_raw_file

    transfer = {'id_raw_file': id_raw_file, 'tag': trans["tag"], 'date_transfer': trans["date_transfer"], 'success': trans["success"], 'target': trans["file"], 'comments': trans["comment"]}
    myobject = dm.database().sqlalchemysession.query(dm.database().tables()['transfer']).filter_by(id_raw_file=id_raw_file, date_transfer=trans["date_transfer"],success=trans["success"]).first()
    if not myobject:
        container = dm.database().tables()['transfer'](**transfer)
        dm.database().sqlalchemysession.add(container)
        dm.database().sqlalchemysession.flush()
        id_raw_file = container.id_raw_file

    dm.database().sqlalchemysession.commit()

dm.database().sqlalchemysession.close()
