#!/sps/grand/software/PythonAppImages/GRANDpython
## =============================================================================
#$ -j n
#$ -notify
#$ -l ct=48:00:00
#$ -l vmem=2.0G
#$ -l fsize=3.0G
#$ -l sps=1
## =============================================================================
import os
import subprocess
import datetime
now=datetime.datetime.now()
with open("GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618.status",'a') as f:
  f.write(str(now)+" Running\n")
wd = os.getcwd()
print(wd)
cmd="/sps/grand/software/Aires-19-04-08-ZHAireS-1.0.30a/aires/bin/ZHAireS<GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618.inp>GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618.stdout"
p = subprocess.Popen(cmd,cwd=wd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
stdout,stderr=p.communicate()
now=datetime.datetime.now()
with open("GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618.status",'a') as f:
  f.write(str(now)+" RunComplete\n")
