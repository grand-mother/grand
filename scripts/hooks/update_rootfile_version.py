#!/usr/bin/env python3

import subprocess
import sys, os

def update_version(repo):
    from os.path import exists
    versionfile = "grand/dataio/version"

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
    #repo.index.add(versionfile)




print("toto")
print(sys.argv)
print(os.environ['GIT_AUTHOR_DATE'])
import git
repo = git.Repo('./')
#print(repo.git.status())
#f.write(repo.git.status())
filenames = (diff_obj.a_path for diff_obj in repo.index.diff('HEAD'))
print(filenames)
for filename in filenames:
    print(filename)
    if filename=="grand/dataio/version":
       print("ACTION")
       update_version(repo)
    else:
        print("Nothing to do")

