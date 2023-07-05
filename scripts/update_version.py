from os.path import exists
versionfile = "grand/io/version"

if not exists(versionfile):
    version = "0.0.0"
else:
    f = open(versionfile, "r")
    version = f.read()
    f.close()

versions = version.split(".")
versions[len(versions) -1] = str(int(versions[len(versions) -1]) + 1)
version = '.'.join(versions)
f = open(versionfile, "w")
f.write(version)
f.close()