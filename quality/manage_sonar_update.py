#! /usr/bin/env python
'''
Created on 22 nov. 2021

@author: jcolley
'''

import os
import sys

print(os.environ["GITHUB_REF_NAME"])
print(os.environ["USER_GIT"])

SHA = os.environ["GITHUB_SHA"]
GITHUB_REF_NAME = os.environ["GITHUB_REF_NAME"]
USER_GIT = os.environ["USER_GIT"][:8]

print(USER_GIT)