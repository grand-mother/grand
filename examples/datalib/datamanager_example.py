import granddb.datamanager as datamanager
dm = datamanager.DataManager('config.ini')
file = "Coarse3.root"
print(dm.get(file))