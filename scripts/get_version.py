from os.path import exists
versionfile = "grand/dataio/version"

if not exists(versionfile):
    version = "0.0.0"
else:
    f = open(versionfile, "r")
    version = f.read()
    f.close()

print("version="+version)
