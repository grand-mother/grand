#! /usr/bin/env python
'''
Created on 22 nov. 2021

@author: jcolley
'''

import os
import sys

USER_SONAR = ['luckyjim', 'rameshkoirala', 'niess', 'grand-oma', "lwpiotr"]

#==================== ENV
SHA = "sha:" + os.environ["GITHUB_SHA"][:8]
BRANCH = os.environ["GITHUB_REF_NAME"]
USER_GIT = os.environ["USER_GIT"]
#LOGIN = "0c9f679e12df438a312b424ae52f5959cd0b5e4a"
LOGIN = os.environ["SONAR_L"]

#==================== FUNCTION
def create_sonar_properties(name="", key="", descr="", login="", n_file='sonar.properties'):
    properties = f'sonar.projectName={name}'
    properties += f'\nsonar.projectKey={key}'
    properties += f'\nsonar.projectDescription={descr}'
    properties += f'\nsonar.login={login}'
    properties += '''
sonar.host.url=https://sonarqube.grand.in2p3.fr

sonar.sources=grand
sonar.sourceEncoding=UTF-8
sonar.python.coverage.reportPaths=quality/report_coverage.xml
sonar.python.pylint.reportPaths=quality/report_pylint.txt
'''
    with open(n_file, 'w') as f_job:
        f_job.write(properties)
        print(n_file)
    print(properties)
    
    
#==================== MAIN
if BRANCH in ['master', 'dev']:
    name = 'grand_'+BRANCH
    create_sonar_properties(name, name, SHA, LOGIN)    
elif USER_GIT in USER_SONAR:
    name = f'user_{USER_GIT}'
    create_sonar_properties(name, name, BRANCH, LOGIN)
else:
    print("No update SonarQube necessary")
sys.exit(0)
    
