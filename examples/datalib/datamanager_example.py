import granddb.datamanager as granddatalib
dm = granddatalib.DataManager('config.ini')
file = "Coarse3.root"
print(dm.get(file))