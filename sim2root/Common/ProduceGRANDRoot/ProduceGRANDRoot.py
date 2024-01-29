#!/usr/env python
#$ -j y
#$ -notify
#$ -l ct=0:30:00
#$ -l vmem=0.5G
#$ -l fsize=0.5G
#$ -l sps=1
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import subprocess
import datetime
import time
import sys
import logging
import os
import glob
#ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
#sys.path.append(ZHAIRESPYTHON)
#PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]


'''
generate grand root from rawroot
'''

#Python=PYTHONINTERPRETER
Python="python"

#Raw2GRAND="../sim2root.py"
Raw2GRAND =  os.path.abspath(os.path.dirname(__file__)) + "/../sim2root.py"


#JobSumbitCommand UserBin InputShowerDirectory Outboxdir/EventName

def CreateGRANDRoot(InputDirectory, OutputDirectory):

  #RENAME THE FILE (not working)
  #head,tail=os.path.split(InputFile)
  #renamed=head+".root"

  #cmd="mv " +  InputFile + " " + renamed
  #print("About to run:"+ cmd)
  #p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  #stdout,stderr=p.communicate()

  rawfile=glob.glob(InputDirectory+"/*.RawRoot")[0]

  if (len(rawfile) <= 4) :
    print("Could not find any RawRoot files in "+InputDirectory+" or we found too many:",rawfile) 
    return -1


  head,tail=os.path.split(rawfile)

  InputJobName=os.path.splitext(tail)[0]

  #python /home/mjtueros/TrabajoTemporario/docker/grand/sim2root/Common/sim2root.py GP300_Xi_Sib_Proton_1.35_65.4_202.3_5388/GP300_Xi_Sib_Proton_1.35_65.4_202.3_5388.RawRoot -e GP300 -s Xi -ru 1 -se 5388 -fo GP300_Xi_Sib_Proton_1.35_65.4_202.3_5388

  EventNumber= InputJobName.split("_")[-1] #the last number of the file neame is the event nuber

  cmd=Python + " " + Raw2GRAND + " " + rawfile + " -fo " + OutputDirectory + " -se " +  EventNumber
  print("About to run:"+ cmd)
  p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
  stdout,stderr=p.communicate()
  print("errors:")
  print(stderr)
  print("output:")
  print(stdout)

  ##################################3


def main():

    logging.basicConfig(level=logging.DEBUG)
    if ( len(sys.argv)<3 ):
        print("""

            Usage: python3  <inputdirectory> <outputdirectory>

        """)
        sys.exit(0)

    inputdirectory = sys.argv[1]
    outputdirectory = sys.argv[2]

    rawfile=glob.glob(inputdirectory+"/*.RawRoot")[0]

    logging.info("About to create GRANDRoot file from "+rawfile)
    logging.debug("output file will be i:"+outputdirectory)

    CreateGRANDRoot(inputdirectory, outputdirectory)

if __name__ == '__main__':
  main()
