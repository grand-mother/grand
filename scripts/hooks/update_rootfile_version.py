#!/usr/bin/env python3

import subprocess
import sys

from os.path import exists
versionfile = "/home/fleg/DEV/GRAND/hooktest"

#if not exists(versionfile):
#    version = "0.0.0"
#else:
#    f = open(versionfile, "r")
#    version = f.read()
#    f.close()

#versions = version.split(".")
#versions[len(versions) -1] = str(int(versions[len(versions) -1]) + 1)
#version = '.'.join(versions)

f = open(versionfile, "w")
#f.write(version)

#print("version="+version)





print("toto")
import git
repo = git.Repo('./')
print(repo.git.status())
(old, new, branch) = sys.stdin.read().split()
print(old)
f.write(old)

print(new)
f.write(new)

print(branch)
f.write(branch)

f.close()