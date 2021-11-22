#! /usr/bin/env python
'''
Created on 22 nov. 2021

@author: jcolley
'''

import os
import sys

import coverage

COV_THRESHOLD = 90

GRAND_ROOT = os.environ["GRAND_ROOT"]
os.chdir(GRAND_ROOT)

cov = coverage.Coverage()
cov.load()

with open(os.devnull, "w") as f:
    total = cov.report(file=f)
    
print("coverage percent: {0:.0f}%".format(total))

if total < COV_THRESHOLD:
    print(f'Coverage percent is failed, threshold is {COV_THRESHOLD}%')
    sys.exit(1)
else:
    print('Coverage successful')
    sys.exit(0)    
