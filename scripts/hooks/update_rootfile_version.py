#!/usr/bin/env python3
# Pre commit script which will update the version number in $versionfile when $watchedfile is commited
# Fleg Sept 2023

versionfile = "grand/dataio/version"
watchedfile = "grand/dataio/root_trees.py"

def update_version(repo):
    from os.path import exists


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
    print("version="+version)
    f.close()
    repo.index.add([versionfile])




import git
repo = git.Repo('./')

filenames = (diff_obj.a_path for diff_obj in repo.index.diff('HEAD'))
if watchedfile in filenames:
    print(watchedfile + " found, updating " + versionfile)
    update_version(repo)


