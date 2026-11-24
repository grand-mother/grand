##
## WARNING WARNING WARNING
##
## this file is a local copy of AiresInfoFunctions from https://github.com/mjtueros/ZHAireS-Python  (on Jun 25 2021)so that 
## you dont have to setup that repository, that requires ZHAireS to be installed. (since we dont use it for now).
##
## Will update it manually if needed. Table exporting will not be available!!
## WARNING WARNING WARNING
##


#if you are having encoding errors, specially in lyon, be sure to force the environment to utf8
#setenv  LANGUAGE en_US.UTF-8 ;setenv  LC_ALL en_US.UTF-8 ;setenv  LANG en_US.UTF-8setenv LC_TYPE en_US.UTF-8
#

#this functions will accept GRAND and AIRES outmode, to give the results in each convention
#it will output the primary zen,azim,energy,primarytype, taken from the .inp file present at input_file_path) (assumed only one .inp file per dir)


#TO DO: treat correctly the different possible primary types, including RASPASS Multi primary
#TO DO: treat correctly the case where a distribution of primary types or energies or angles is put in the input
#TO DO: Check consistency of the output (energy within a range, angles within a range, etc)
#TO DO: have the .sry reader regenerate the summary using AiresSry, if the file is not foun

#6/2019 Matias Tueros, first attempt at python based on original script by Anne Zilles.
#10/2019 Migrated them to make it the official library
#12/2019 Set up on git
#03/2020 Corona Virus
#05/2023 RawRoot work

import sys
from sys import argv
import os

import glob
import logging

logging.basicConfig(level=logging.DEBUG)

#this function reads he Aires .sry file and tries to extract information from the simulation parameters
#using the .sry file is preferred to using the .inp files because:
#Output is standarized: you dont know what you can find in an .inp file (leading spaces, commented lines, repeated keywords, etc)
#Output is what really happened, not what the user whished it would happen when he did his crappy .inp file.

#lets break the ReadAiresSry into smaller modules. It has the disadventage of opening, scanning and closing the file each time
#but it adds modularity, and closes the files when it does not use it any more.
#If at some point we need speed, then we could input datafile, and make a wraper for opening the file

#AiresPath="/home/mjtueros/aires/bin"
#print(os.environ)
try:
  AiresPath=os.environ["AIRESBINDIR"]
except:
  AiresPath="."

print("AiresPath:",AiresPath)
def GetZenithAngleFromSry(sry_file,outmode="GRAND"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Primary zenith angle:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          zen=float(stripedline[len(stripedline)-2])
          if outmode == 'GRAND':
            zen = 180-zen  #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
          #logging.debug('Found Zenith ' + str(zen))
          return zen
      try:
        zen
      except NameError:
        zen = 0 #If no zenith angle was included in the input file, AIRES defaults to 0
        if outmode == 'GRAND':
          zen = 180-0 #that translates to 180 in GRAND
        logging.info("Zenith Angle not found in sry file, defaulting to:" + str(zen))
        return zen
  except:
    logging.error("GetZenithAngleFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetAzimuthAngleFromSry(sry_file,outmode="GRAND"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Primary azimuth angle:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          azim = float(stripedline[len(stripedline)-2])
          if outmode == 'GRAND':
            azim=azim+180 #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
            if azim>=360:
              azim= azim-360
          #logging.debug('Found Azimuth ' + str(azim))
          return azim
      try:
        azim
      except NameError:
        azim = 0 #If no azimuth angle was included in the input file, AIRES defaults to 0
        if outmode == 'GRAND':
          azim = 0+180 # that translates to 18
        logging.info("Azimuth Angle not found in sry file, defaulting to:" + str(azim))
        return azim

  except:
    logging.error("GetAzimuthAngleFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetEnergyFromSry(sry_file,outmode="GRAND"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Primary energy:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          try: #this is to detect if it is the first time we find the string, so that we ignore the following
            energy
          except NameError:
            energy = float(stripedline[len(stripedline)-2])
            unit= str(stripedline[len(stripedline)-1])
            if unit == "eV\n":
             energy = energy *1e-9
            if unit == "KeV\n":
             energy = energy *1e-6
            if unit == "MeV\n":
              energy = energy *1e-3
            if unit == "GeV\n":
              energy = energy
            if unit == "TeV\n":
              energy = energy *1e3
            if unit == "PeV\n":
              energy = energy *1e6
            if unit == "EeV\n":
              energy = energy *1e9

            if outmode == 'GRAND': #AIRES mode outputs in GeV, GRAND in EeV
              energy = energy * 1e-9
            #logging.debug('Found Energy ' + str(energy)) #debug level 1
            return energy
      try:
        energy
      except NameError:
        logging.error('warning energy not found, Aires has no default value,  cannot continue')
        exit()
  except:
    logging.error("GetEnergyFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


#output is in meters
def GetCorePositionFromInp(inp_file,outmode="N/A"):
  try:
    datafile=open(inp_file,'r')
    with open(inp_file, "r") as datafile:
      for line in datafile:
        if '#Core Position:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.split(' ',-1)
          x=float(stripedline[1])
          y=float(stripedline[2])
          z=float(stripedline[3])
          coreposition=(x,y,z)
          return coreposition
      try:
        coreposition
      except NameError:
        logging.error('warning core position not found, defaulting to (0,0,0)')
        return (0.0,0.0,0.0)
  except:
    logging.error("GetCorePositionFromInp:file not found or invalid:"+inp_file)
    raise
    return -1

def GetThinningRelativeEnergyFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Thinning energy:' in line:
          line = line.lstrip()
          stripedline=line.split('Thinning energy:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          Thinning=stripedline[0]
          Relative=stripedline[1]
          if(Relative!="Relative\n"):
             logging.error('warning , we only support Relative thinning for now, sorry!. This is easy to implement if you need it!')
             return -1
          return float(Thinning)
      try:
        Thinning
      except NameError:
        logging.error('warning Thinning energy not found, Aires has no default value, cannot continue')
        exit()
  except:
    logging.error("GetThinningRelativeEnergyFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetGammaEnergyCutFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Cut energy for gammas:' in line:
          line = line.lstrip()
          stripedline=line.split('Cut energy for gammas:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energy=stripedline[0]
          unit=stripedline[1]
          if unit == "eV\n":
            energy = energy *1e-6
          if unit == "KeV\n":
            energy = energy *1e-3
          if unit == "MeV\n":
            energy = energy
          if unit == "GeV\n":
            energy = energy *1e3
          if unit == "TeV\n":
            energy = energy *1e6
          if unit == "PeV\n":
            energy = energy *1e9
          if unit == "EeV\n":
            energy = energy *1e12
          return energy
      try:
        energy
      except NameError:
        logging.error('warning GammaCutEnergy not found, defaulting to 80keV')
        return 80+1e-3
  except:
    logging.error("GetGammaEnergyCutFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetElectronEnergyCutFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Cut energy for e+ e-:' in line:
          line = line.lstrip()
          stripedline=line.split('Cut energy for e+ e-:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energy=stripedline[0]
          unit=stripedline[1]
          if unit == "eV\n":
            energy = energy *1e-6
          if unit == "KeV\n":
            energy = energy *1e-3
          if unit == "MeV\n":
            energy = energy
          if unit == "GeV\n":
            energy = energy *1e3
          if unit == "TeV\n":
            energy = energy *1e6
          if unit == "PeV\n":
            energy = energy *1e9
          if unit == "EeV\n":
            energy = energy *1e12
          return energy
      try:
        energy
      except NameError:
        logging.error('warning ElectronCutEnergy not found, defaulting to 80keV')
        return 80+1e-3
  except:
    logging.error("GetElectronEnergyCutFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetMuonEnergyCutFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Cut energy for mu+ mu-:' in line:
          line = line.lstrip()
          stripedline=line.split('Cut energy for mu+ mu-:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energy=stripedline[0]
          unit=stripedline[1]
          if unit == "eV\n":
            energy = energy *1e-6
          if unit == "KeV\n":
            energy = energy *1e-3
          if unit == "MeV\n":
            energy = energy
          if unit == "GeV\n":
            energy = energy *1e3
          if unit == "TeV\n":
            energy = energy *1e6
          if unit == "PeV\n":
            energy = energy *1e9
          if unit == "EeV\n":
            energy = energy *1e12
          return energy
      try:
        energy
      except NameError:
        logging.error('warning MuonCutEnergy not found, defaulting to 10MeV')
        return 10
  except:
    logging.error("GetMuonEnergyCutFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetMesonEnergyCutFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Cut energy for mesons:' in line:
          line = line.lstrip()
          stripedline=line.split('Cut energy for mesons:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energy=stripedline[0]
          unit=stripedline[1]
          if unit == "eV\n":
            energy = energy *1e-6
          if unit == "KeV\n":
            energy = energy *1e-3
          if unit == "MeV\n":
            energy = energy
          if unit == "GeV\n":
            energy = energy *1e3
          if unit == "TeV\n":
            energy = energy *1e6
          if unit == "PeV\n":
            energy = energy *1e9
          if unit == "EeV\n":
            energy = energy *1e12
          return energy
      try:
        energy
      except NameError:
        logging.error('warning MesonCutEnergy not found, defaulting to 60MeV')
        return 60
  except:
    logging.error("GetMesonEnergyCutFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetNucleonEnergyCutFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Cut energy for nucleons:' in line:
          line = line.lstrip()
          stripedline=line.split('Cut energy for nucleons:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energy=stripedline[0]
          unit=stripedline[1]
          if unit == "eV\n":
            energy = energy *1e-6
          if unit == "KeV\n":
            energy = energy *1e-3
          if unit == "MeV\n":
            energy = energy
          if unit == "GeV\n":
            energy = energy *1e3
          if unit == "TeV\n":
            energy = energy *1e6
          if unit == "PeV\n":
            energy = energy *1e9
          if unit == "EeV\n":
            energy = energy *1e12
          return energy
      try:
        energy
      except NameError:
        logging.error('warning NucleonCutEnergy not found, defaulting to 60MeV')
        return 60
  except:
    logging.error("GetNucleonEnergyCutFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetPrimaryFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Primary particle:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          if(len(stripedline)==3):
            primarytype = str(stripedline[len(stripedline)-1])
          elif(len(stripedline)==5):
            primarytype = str(stripedline[len(stripedline)-3])
          elif(len(stripedline)==6):
            primarytype = str(stripedline[len(stripedline)-4])
          elif(len(stripedline)==7):
            primarytype = str(stripedline[len(stripedline)-5])
          else:
            primarytype = "unknown"
          primarytype=primarytype.replace('\n','')
          #logging.debug('Found Primary ' + primarytype) #debug level 1

          if(outmode=="GRAND"):
           #convert to GRAND coding. PDG?
            if primarytype=="Proton":
              primarytype=2212
            if primarytype=="Neutron":
              primarytype=2112
            if primarytype=="Pi0":
              primarytype=111
            if primarytype=="Pi+":
              primarytype=211
            if primarytype=="Pi-":
              primarytype=-211

          return primarytype
      try:
        primarytype
      except NameError:
        logging.error('warning primary not found, Aires has no default value, cannot continue')
        exit()
  except:
    logging.error("GetPrimaryFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetSlantXmaxFromSry(sry_file,outmode="N/A"): #To do. Handle when Xmax is not found, becouse the fit didnt converge
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Sl. depth of max. (g/cm2)' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          xmax = float(stripedline[len(stripedline)-1])
          #logging.debug('Found Xmax ' + str(xmax)) #debug level 1
          return xmax
      try:
        xmax
      except NameError:
        logging.info('warning xmax not found')
        xmax=-1
        return xmax
  except:
    logging.error("GetSlantXmaxFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
#
#Charged pcles. at maximum
def GetNmaxFromSry(sry_file,outmode="N/A"): #To do. Handle when Xmax is not found, becouse the fit didnt converge
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Charged pcles. at maximum' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          nmax = float(stripedline[len(stripedline)-1])
          #logging.debug('Found Nmax ' + str(nmax)) #debug level 1
          return nmax
      try:
        xmax
      except NameError:
        logging.info('warning nmax not found')
        xmax=-1
        return xmax
  except:
    logging.error("GetNmaxFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
#                              Altitude  Distance     x        y        z
#      Location of max.(Km):     3.612     3.66     0.00     0.57     3.61
def GetKmXmaxFromSry(sry_file,outmode="N/A"): #To do. Handle when Xmax is not found, becouse the fit didnt converge, or becouse this is not ZHAireS, or its latest version
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if ('Location of max.(Km)' in line) or ('Pos. Max.' in line)  :
          line = line.lstrip()
          stripedline=line.split()
          kmxmax = float(stripedline[len(stripedline)-5])
          distance = float(stripedline[len(stripedline)-4])
          x = float(stripedline[len(stripedline)-3])
          y = float(stripedline[len(stripedline)-2])
          z = float(stripedline[len(stripedline)-1])
          logging.debug("Found Xmax altitude " + str(kmxmax) + " distance " + str(distance) + " x:" + str(x) + " y:"+ str(y) + " z:" + str(z) ) #debug level 1
          return kmxmax,distance,x,y,z
      try:
        kmxmax
      except NameError:
        logging.info('warning distance to xmax not found')
        return -1,-1,-1,-1,-1
  except:
    logging.error("GetKmXmaxFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetExpectedKmXmaxFromSry(sry_file,outmode="N/A"): #To do. Handle when Xmax is not found, becouse the fit didnt converge, or becouse this is not ZHAireS, or its latest version
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Exp Distance To Xmax' in line:
          line = line.lstrip()
          stripedline=line.split()
          kmxmax = float(stripedline[len(stripedline)-2])
          logging.debug("Found Expected Xmax distance " + str(kmxmax) ) #debug level 1
          return kmxmax
      try:
        kmxmax
      except NameError:
        logging.info('warning expected distance to xmax not found')
        return -1
  except:
    logging.error("GetExpectedKmXmaxFromSry:file not found or invalid:"+sry_file)
    raise
    return -1




def GetTaskNameFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Task Name:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          taskname=stripedline[len(stripedline)-1]
          taskname=taskname.replace('\n','')
          #logging.debug("Found taskname " + taskname)
          if '...' in taskname:
            base=os.path.basename(sry_file)
            taskname=os.path.splitext(base)[0]
            logging.debug("taskname contained ... (too long), using filename instead")
            print("taskname:"+taskname)

          return taskname
      try:
        taskname
      except NameError:
        logging.error('warning taskname not found, Aires has no default value, cannot continue')
        exit()
  except:
    logging.error("GetTaskNameFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetRandomSeedFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Seed of random generator:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          randomseed=stripedline[len(stripedline)-1]
          randomseed=randomseed.replace('\n','')
          return randomseed
      try:
        randomseed
      except NameError:
        logging.error('warning randomseed not found, Aires has no default value, cannot continue')
        exit()
  except:
    logging.error("GetRandomSeedFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

#output is in meters
def GetGroundAltitudeFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Ground altitude:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.split(' ',-1)
          groundalt=float(stripedline[1])
          unit=stripedline[2]
          if unit == "km":
           groundalt=groundalt*1000.0
          if unit == "cm":
           groundalt=groundalt/100.0
          return groundalt
      try:
        groundalt
      except NameError:
        logging.error('warning groundalt not found, defaulting to sea level')
        return 0
  except:
    logging.error("GetGroundAltitudeFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
    
def GetGroundDepthFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Ground altitude:' in line:
          start_index = line.find("(") + 1
          #catch the case where is the default indicator "(D)" at the begining of the linde
          if(line[start_index]=="D"):
            line=line[start_index:]
            start_index = line.find("(") + 1          
          end_index = line.find(" ", start_index)
          depth_str = line[start_index:end_index]
          try:
           depth = float(depth_str)
           return depth
          except ValueError:
            # Handle the case where the extracted string cannot be converted to a float
            logging.error("Could not convert string to depth value"+str(depth_str))
            return None          
      try:
        depth
      except NameError:
        logging.error('warning depth not found, defaulting to 1035 g/cm2')
        return 1035
  except:
    logging.error("GetGroundDepthFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
       
def GetTimeBinFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Time Domain Bin Size:' in line:
          line = line.lstrip()
          stripedline=line.split('Time Domain Bin Size:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          timebin=float(stripedline[0])
          unit=stripedline[1]
          if unit != "sec\n":
              logging.error('warning, time bin must be in seconds on the summary file, other units not supported yet')
              return -1
          return timebin*1e9

      try:
        timebin
      except NameError:
        logging.error('warning timebin not found')
        return 0
  except:
    logging.error("GetTimeBinFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetTimeWindowMinFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Antenna Time Window Min:' in line:
          line = line.lstrip()
          stripedline=line.split('Antenna Time Window Min:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          timemin=float(stripedline[0])
          unit=stripedline[1]
          if unit != "sec\n":
              logging.error('warning, time window must be in seconds on the summary file, other units not supported yet')
              return -1
          return timemin*1e9

      try:
        timemin
      except NameError:
        logging.error('warning timewindow min not found')
        return 0
  except:
    logging.error("GetTimeWindowMinFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetTimeWindowMaxFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Antenna Time Window Max:' in line:
          line = line.lstrip()
          stripedline=line.split('Antenna Time Window Max:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          timemax=float(stripedline[0])
          unit=stripedline[1]
          if unit != "sec\n":
              logging.error('warning, time window must be in seconds on the summary file, other units not supported yet')
              return -1
          return timemax*1e9

      try:
        timemax
      except NameError:
        logging.error('warning timewindow max not found')
        return 0
  except:
    logging.error("GetTimeWindowMaxFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetWeightFactorFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Max. stat. weight factor:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          wf=float(stripedline[0])
          return wf
      try:
        wf
      except NameError:
        logging.info('warning weight factor not found, default is 12')
        return 12.0
  except:
    logging.error("GetWeightFactorFromSry:file not found or invalid:"+sry_file)
    raise
    return -1



def GetEMtoHadronWFRatioFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'EM to Hadron WF ratio:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          wf=float(stripedline[0])
          return wf
      try:
        wf
      except NameError:
        logging.info('warning EMtoHadronWFRatio not found, default is 88')
        return 88.0
  except:
    logging.error("GetEMtoHadronWFRatioFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


#From Aires Manual sec 2.3.2 Aires Extended Thinning algorithm
def ComputeMaxWeight(PrimaryEnergyInGeV,RelativeThinning,WeightFactor):
   MaxWeight=PrimaryEnergyInGeV*RelativeThinning*WeightFactor/14.0
   return MaxWeight


def GetMagneticFieldFromSry(sry_file,outmode="N/A"):
  fieldintensity=float(0.0)
  fieldinclination=float(0.0)
  fielddeclination=float(0.0)
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Geomagnetic field:' in line:
          line = line.lstrip()
          if 'Off'in line:
            fieldintensity=0.0
            fieldinclination=0.0
            fielddeclination=0.0
          else:
            stripedline=line.split('Intensity:',-1)
            intensityline=stripedline[1]
            intensityline=intensityline.lstrip()
            stripedline=intensityline.split(' ',-1)
            fieldintensity=float(stripedline[0])
        if 'I:' in line:
          line = line.lstrip()
          stripedline=line.split('I:',-1)
          inclinationline=stripedline[1]
          inclinationline=inclinationline.lstrip()
          stripedline=inclinationline.split(' ',-1)
          fieldinclination=float(stripedline[0])
          stripedline=line.split('D:',-1)
          declinationline=stripedline[1]
          declinationline=declinationline.lstrip()
          stripedline=declinationline.split(' ',-1)
          fielddeclination=float(stripedline[0])

      try:
        fieldintensity
        return fieldintensity,fieldinclination,fielddeclination
      except NameError:
        logging.error('warning MagneticField not found, defaulting to sea level')
        return 0,0,0
  except:
    logging.error("GetMagneticFieldFromSry:file not found or invalid:"+sry_file)
    raise
    return -1,-1,-1


def GetTotalCPUTimeFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Total CPU time' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          CPUtime=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          CPUtime=CPUtime.replace('\n','')
          stripedline=CPUtime.split(' ',-1)
          time=0.0
          if(len(stripedline)>1):
            if(stripedline[-1]=="sec"):
              time=time+float(stripedline[-2])
            if(stripedline[-1]=="min"):
              time=time+60.0*int(stripedline[-2])
            if(stripedline[-1]=="hr"):
              time=time+60.0*60.0*int(stripedline[-2])
          if(len(stripedline)>3):
            if(stripedline[-3]=="sec"):
              time=time+float(stripedline[-4])
            if(stripedline[-3]=="min"):
              time=time+60.0*int(stripedline[-4])
            if(stripedline[-3]=="hr"):
              time=time+60.0*60.0*int(stripedline[-4])
          if(len(stripedline)>5):
            if(stripedline[-5]=="sec"):
              time=time + float(stripedline[-6])
            if(stripedline[-5]=="min"):
              time=time + 60.0*int(stripedline[-6])
            if(stripedline[-5]=="hr"):
              time=time + 60.0*60.0*int(stripedline[-6])
          CPUtime=time
          #print("CPUTime:",CPUtime)
          return CPUtime
      try:
        CPUtime
      except NameError:
        logging.error('warning Total CPU time not found')
        return -1
  except:
    logging.error("GetTotalCPUTimeFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetHadronicModelFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Hadronic Mean Free Paths' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          HadronicModel=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          return HadronicModel
      try:
        HadronicModel
      except NameError:
        logging.error('warning Hadronic Model not found')
        return -1
  except:
    logging.error("GetHadronicModelFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetRandomSeedFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Seed of random generator:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          RandomSeed=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          return RandomSeed
      try:
        RandomSeedl
      except NameError:
        logging.error('warning RandomSeed not found')
        return -1
  except:
    logging.error("GetRandomSeedFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetAiresVersionFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'This is AIRES version' in line:
          line = line.lstrip()
          stripedline=line.split('This is AIRES version',-1)
          stripedline=stripedline[-1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          AiresVersion=stripedline[0]
          #ogging.debug("Found Version " + AiresVersion)
          return AiresVersion
      try:
        AiresVersion
      except NameError:
        logging.error('warning Aires Version not found')
        return -1
  except:
    logging.error("GetAiresVersionFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetZHAireSVersionFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'With ZHAireS version' in line:
          line = line.lstrip()
          stripedline=line.split('With ZHAireS version',-1)
          stripedline=stripedline[-1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          ZHAiresVersion=stripedline[0]
          #ogging.debug("Found Version " + AiresVersion)
          return ZHAiresVersion
      try:
        ZHAiresVersion
      except NameError:
        logging.error('warning ZHAireS Version not found')
        return -1
  except:
    logging.error("GetZHAireSVersionFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetAtmosphericModelFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Atmospheric model' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          AtmosphericModel=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          return AtmosphericModel
      try:
        AtmosphericModel
      except NameError:
        logging.error('warning Atmospheric Model not found')
        return -1
  except:
    logging.error("GetAtmosphericModelFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetRefractionIndexModelFromSry(sry_file,outmode="N/A"): #for aires 19.04.06
  try:
    datafile=open(sry_file,'r')
    datfile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Glasgow-Dale refraction index model' in line:
          IndexModel="Glasgow-Dale"
          return IndexModel
        if 'Exponential refraction index model' in line:  
          IndexModel="Exponential"
          return IndexModel
        if 'Constant refraction index model':
          IndexModel="Constant"
          return IndexModel  
      try:
        IndexModel
      except NameError:
        logging.error('warning Refraction Index Model not found')
        return -1
  except:
    logging.error("GetRefractionIndexModelFromSry:file not found or invalid:"+sry_file)
    raise
    return -1    
def GetSiteFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Site:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          Site=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          return Site
      try:
        Site
      except NameError:
        logging.error('warning Site not found')
        return -1
  except:
    logging.error("GetSiteFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetLatLongFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if '(Lat:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          Lat= stripedline[1]
          Lat= Lat.lstrip()
          Lat= Lat.split(" ",-1)
          Lat= Lat[0]
          Long=stripedline[2]
          Long= Long.lstrip()
          Long= Long.split(" ",-1)
          Long= Long[0]
          return Lat,Long
      try:
        Lat
        Long
      except NameError:
        logging.error('warning Latitude or Longitude not found')
        return -1,-1
  except:
    logging.error("GetLatLongFromSry:file not found or invalid:"+sry_file)
    raise
    return -1,-1

def GetDateFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Date:' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          Date=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
          return Date
      try:
        Date
      except NameError:
        logging.error('warning Date not found')
        return -1
  except:
    logging.error("GetDateFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetInjectionAltitudeFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Injection altitude' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1) #since somtimes we can have the (D) of default and sometimes not i make sure this is whatever is after the :
          stripedline=stripedline[1]
          stripedline=stripedline.split(' ',-1)
          injalt=float(stripedline[1])
          unit=stripedline[2]
          if unit == "km":
           injalt=injalt*1000.0
          if unit == "cm":
           injalt=injalt/100.0
          return injalt
      try:
        injalt
      except NameError:
        logging.error('warning Injection Altitude not found')
        return -1
  except:
    logging.error("GetInjectionAltitudeFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetEnergyFractionInNeutrinosFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if 'Neutrinos:' in line:
          if(founditonce==1):
            line = line.lstrip()
            stripedline=line.split('Neutrinos:',-1)
            stripedline=stripedline[1]
            stripedline=stripedline.lstrip()
            stripedline=stripedline.split(' ',-1)
            energyf=float(stripedline[0])
            return energyf
          founditonce=founditonce+1
      try:
        energyf
      except NameError:
        logging.error('warning NeutrinoEnergyFraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionInNeutrinosFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetEnergyFractionDepositedFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if "PER SHOWER BALANCE OF ENERGY" in line: #we are in the correct section
          founditonce=1
        if founditonce==1 and 'Totals:' in line:
          founditonce=2
        if founditonce==2 and "Medium losses:" in line: 
          line = line.lstrip()
          stripedline=line.split('Medium losses:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energyf=float(stripedline[0])
          return energyf
      try:
        energyf
      except NameError:
        logging.error('GetEnergyFractionDepositedFromSry: Deposited Energy Fraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionDepositedFromSry:file not found or invalid:"+sry_file)
    raise
    return -1    

def GetEnergyFractionLostFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if "PER SHOWER BALANCE OF ENERGY" in line : #we are in the correct section
          founditonce=1
        if founditonce==1 and ('Totals:' in line) :
          founditonce=2
        if founditonce==2 and ("Particles lost:" in line):
          line = line.lstrip()
          stripedline=line.split('Particles lost:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energyf=float(stripedline[0])
          return energyf
      try:
        energyf
      except NameError:
        logging.error('GetEnergyFractionLostFromSry: Lost Energy Fraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionLostFromSry:file not found or invalid:"+sry_file)
    raise
    return -1


def GetEnergyFractionLowFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if "PER SHOWER BALANCE OF ENERGY" in line : #we are in the correct section
          founditonce=1
        if founditonce==1 and ('Totals:' in line) :
          founditonce=2
        if founditonce==2 and ("Low-E particles:" in line):
          line = line.lstrip()
          stripedline=line.split('Low-E particles:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energyf=float(stripedline[0])
          return energyf
      try:
        energyf
      except NameError:
        logging.error('GetEnergyFractionLowFromSry: Low Energy Fraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionLowFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetEnergyFractionGroundFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if "PER SHOWER BALANCE OF ENERGY" in line : #we are in the correct section
          founditonce=1
        if founditonce==1 and ('Totals:' in line) :
          founditonce=2
        if founditonce==2 and ("Pcles. that reached gnd.:" in line):
          line = line.lstrip()
          stripedline=line.split('Pcles. that reached gnd.:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energyf=float(stripedline[0])
          return energyf
      try:
        energyf
      except NameError:
        logging.error('GetEnergyFractionGroundFromSry: Ground Energy Fraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionGroundFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
        
def GetEnergyFractionUnphysFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      founditonce=0
      for line in datafile:
        if "PER SHOWER BALANCE OF ENERGY" in line : #we are in the correct section
          founditonce=1
        if founditonce==1 and ('Totals:' in line) :
          founditonce=2
        if founditonce==2 and ("Unphysical particles:" in line):
          line = line.lstrip()
          stripedline=line.split('Unphysical particles:',-1)
          stripedline=stripedline[1]
          stripedline=stripedline.lstrip()
          stripedline=stripedline.split(' ',-1)
          energyf=float(stripedline[0])
          return energyf
      try:
        energyf
      except NameError:
        logging.error('GetEnergyFractionUnphysFromSry: Unphysical Energy Fraction not found')
        return -1
  except:
    logging.error("GetEnergyFractionUnphysFromSry:file not found or invalid:"+sry_file)
    raise
    return -1
def get_antenna_t0(xant,yant,hant, azimuthdeg, zenithdeg):
    #this code is copied from zhaires fieldinit
    #returns the t0 of the antenna in ns
    #x,y,hant is antenna position in zhaires reference frame, hant is the altitude above ground (its making flat earth asumptions for now)
    #azimuth and zenith are in ZHAireS,degrees
    cspeed = 299792458.0
    #get incoming azimut in radians
    phidirrad=azimuthdeg*np.pi/180.0
    #set phidirrad in the rangle [0,2pi]
    if(phidirrad < 0.0):
       phidirrad=phidirrad+2.0*np.pi

    zenithrad=zenithdeg*np.pi/180.0
    # Auxiliary variables
    coszenith=np.cos(zenithrad)
    sinzenith=np.sin(zenithrad)
    tanzenith=np.tan(zenithrad)


    # distance to the axis
    Rant=np.sqrt(xant*xant+yant*yant)
    #phi
    phiantrad=np.arctan2(yant,xant)
    if(phiantrad  < 0):
      phiantrad=2.0*np.pi+phiantrad

    #Adjusting time window
    #phi of antenna
    #projection of r of antenna on shower phi direction
    angle=np.absolute(phidirrad-phiantrad)

    rproj=Rant*np.cos(angle)

    dtna=(hant/coszenith + (rproj-hant*tanzenith)*sinzenith)/cspeed

    return dtna*1.0e9

def GetAntennaInfoFromSry(sry_file,outmode="N/A"):
  AntennaOrder=[]
  AntennaID=[]
  AntennaX=[]
  AntennaY=[]
  AntennaZ=[]
  AntennaT=[]
  Read=False
  ReadLegacy=False
  AntennaN=0
  try:
    datafile=open(sry_file,'r')
    datafile.close()
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if(Read):
          stripedline=line.split()
          if(len(stripedline)==6):
            AntennaOrder.append(stripedline[0])
            AntennaID.append(stripedline[1])
            AntennaX.append(stripedline[2])
            AntennaY.append(stripedline[3])
            AntennaZ.append(stripedline[4])
            if(stripedline[5]=="**********"):
              print("trying to recover an antenna t0")
              azimuthdeg=GetAzimuthAngleFromSry(sry_file,outmode="AIRES")
              zenithdeg=GetZenithAngleFromSry(sry_file,outmode="AIRES")
              xant=float(stripedline[2])
              yant=float(stripedline[3])
              zant=float(stripedline[4])
              ground=GetGroundAltitudeFromSry(sry_file)
              hant=zant-ground
              print(xant,yant,zant,hant,azimuthdeg,zenithdeg)
              stripedline[5]=str(get_antenna_t0(xant,yant,hant, azimuthdeg, zenithdeg))
              print(stripedline[5])


            AntennaT.append(stripedline[5])
          else:
            Read=False

            #now, i need to make the AntennaID Unique, so that i can store them in the file
            dups = {}

            for i, val in enumerate(AntennaID):
                if val not in dups:
                    # Store index of first occurrence and occurrence value
                    dups[val] = [i, 1]
                else:
                    # Special case for first occurrence
                    if dups[val][1] == 1:
                        AntennaID[dups[val][0]] += str(dups[val][1])

                    # Increment occurrence value, index value doesn't matter anymore
                    dups[val][1] += 1

                    # Use stored occurrence value
                    AntennaID[i] += str(dups[val][1])
                    
            return AntennaOrder,AntennaID,AntennaX,AntennaY,AntennaZ,AntennaT

        if(ReadLegacy):
          stripedline=line.split()
          if(len(stripedline)==5):
            AntennaX.append(stripedline[1])
            AntennaY.append(stripedline[2])
            AntennaZ.append(stripedline[3])
            AntennaT.append(stripedline[4])
            AntennaID.append("Antenna"+str(AntennaN))
            AntennaN=AntennaN+1
          else:
            ReadLegacy=False
            return AntennaID,AntennaX,AntennaY,AntennaZ,AntennaT

        elif 'Antenna|      Label      |' in line:
          Read=True
        elif 'Antenna|   X [m]' in line:
          ReadLegacy=True

  except:
    logging.error("GetAntennaInfoFromSry:file not found or invalid:"+sry_file)
    raise
    return -1,-1,-1,-1,-1

def GetLongitudinalTable(Path,TableNumber,Slant=True,Precision="Double",TaskName="Unknown"):
    #todo: check against a list of valid tables
    deletefile=False
    
    if(TableNumber<1000):
      TableNumber="0"+str(TableNumber)
    else:
      TableNumber=str(TableNumber)
      
      
    
    if(TaskName=="Unknown"):
      sryfile=glob.glob(Path+"/*.sry")
      idffile=glob.glob(Path+"/*.idf")
      inpfile=glob.glob(Path+"/*.inp")
      tablefile=glob.glob(Path+"/*.t"+TableNumber)

      if(len(tablefile)==0 and len(idffile)==0):
        logging.error("The requested table was not found, and the idf file is not present. Cannot get the table "+str(TableNumber))
        return -1
    else :
      sryfile=[Path+"/"+TaskName+".sry"]
      idffile=[Path+"/"+TaskName+".idf"]
      inpfile=[Path+"/"+TaskName+".inp"]
      tablefile=glob.glob(Path+"/"+TaskName+".t"+TableNumber)      
          

    if(len(idffile)==1 and len(tablefile)==0):
      logging.debug("GetLongitudinalTable could not find the "+str(TableNumber)+" table, trying to get it from the idf")
      base=os.path.basename(idffile[0])
      taskname=os.path.splitext(base)[0]

      if(len(Path)+len(taskname)<60 and len(taskname)<60):
        if(Slant==True):
            cmd=AiresPath+"/AiresExport -O a "+Path+"/"+taskname+" "+TableNumber + " >/dev/null"
        elif(Slant==False):
            cmd=AiresPath+"/AiresExport "+Path+"/"+taskname+" "+TableNumber + " >/dev/null"
        else:
            logging.error("unrecognized Slant value, please state either True/False")
            return -1
        logging.debug("abut to run "+ cmd)
        #print("1:",cmd)
        os.system(cmd)
        tablefile=glob.glob(Path+"/*.t"+TableNumber)

        if(len(tablefile)==1):
            logging.debug("Table exported successfully")
            deletefile=True

      elif(len(taskname)<60):
        logging.debug("idf path+name is too long, need to copy the idf here")
        cmd="cp "+idffile[0]+" ."
        os.system(cmd)

        if(Slant==True):
            cmd=AiresPath+"/AiresExport -O a "+taskname+" "+TableNumber + " >/dev/null"
        elif(Slant==False):
            cmd=AiresPath+"/AiresExport "+taskname+" "+TableNumber + " >/dev/null"
        else:
            logging.error("unrecognized Slant value, please state either True/False")
            return -1
        #print("2:",cmd)
        os.system(cmd)
        tablefile=glob.glob("*.t"+TableNumber)
        logging.debug(tablefile)

        if(len(tablefile)==1):#this means we are not wo
            logging.debug("Table exported successfully")
            deletefile=True
            cmd="rm "+taskname+".idf"
            os.system(cmd)
            cmd="rm "+taskname+".lgf"
            os.system(cmd)
      else:
       logging.error("task name is to long, AIRES does not support that!")

    if(len(tablefile)==1):
      logging.debug("reading file")

      from numpy import loadtxt

      if(Precision=="Double"):
        if(TableNumber!="0100"):
          numpyarray=loadtxt(tablefile[0],usecols=(1,2),dtype='f8')
        else:
          numpyarray=loadtxt(tablefile[0],usecols=(1,2,3),dtype='f8')           
      elif(Precision=="Simple"):
        if(TableNumber!="0100"):
          numpyarray=loadtxt(tablefile[0],usecols=(1,2),dtype='f4')
        else:
          numpyarray=loadtxt(tablefile[0],usecols=(1,2,3),dtype='f4')        
      else:
        logging.error("unrecognized precison, please state either Double or Simple")
        return -1

      if(deletefile==True):
        cmd="rm "+tablefile[0]
        os.system(cmd)
        logging.debug("Table deleted successfully")

      return numpyarray
    else:
      logging.error("The requested table was not found and could not be regenerated. Sorry")
      return -1


def GetLateralTable(Path,TableNumber,Density=True,Precision="Double"):
    #todo: check against a list of valid tables
    deletefile=False
    sryfile=glob.glob(Path+"/*.sry")
    idffile=glob.glob(Path+"/*.idf")
    inpfile=glob.glob(Path+"/*.inp")
    tablefile=glob.glob(Path+"/*.t"+str(TableNumber))

    if(len(tablefile)==0 and len(idffile)==0):
      logging.error("LDF.The requested table was not found, and the idf file is not present. Cannot get the table")
      return -1

    if(len(idffile)==1 and len(tablefile)==0):
      logging.debug("LDF.could not find the table, trying to guess it from the idf")
      base=os.path.basename(idffile[0])
      taskname=os.path.splitext(base)[0]

      if(len(Path)+len(taskname)<60 and len(taskname)<60):

          if(Density==True):
            cmd=AiresPath+"/AiresExport -O dX "+Path+"/"+taskname+" "+str(TableNumber)
          elif(Density==False):
            cmd=AiresPath+"/AiresExport -O X "+Path+"/"+taskname+" "+str(TableNumber)
          else:
            logging.error("LDF.unrecognized Density value, please state either True/False")
            return -1

          os.system(cmd)
          tablefile=glob.glob(Path+"/*.t"+str(TableNumber))

          if(len(tablefile)==1):
            logging.debug("LDF.Table exported successfully")
            deletefile=True

      elif(len(taskname)<60):
          logging.debug("LDF. idf path+name is too long, need to copy the idf here")
          cmd="cp "+idffile[0]+" ."
          os.system(cmd)

          if(Density==True):
            cmd=AiresPath+"/AiresExport -O dX "+taskname+" "+str(TableNumber)
          elif(Density==False):
            cmd=AiresPath+"/AiresExport -O X "+taskname+" "+str(TableNumber)
          else:
            logging.error("LFD.unrecognized Density value, please state either True/False")
            return -1

          os.system(cmd)
          tablefile=glob.glob("*.t"+str(TableNumber))

          if(len(tablefile)==1):
            logging.debug("LDF.Table exported successfully")
            deletefile=True
            cmd="rm "+taskname+".idf"
            os.system(cmd)
            cmd="rm "+taskname+".lgf"
            os.system(cmd)

      else:
         logging.debug("LDF.task name is to long, AIRES does not support that!")

    if(len(tablefile)==1):
      logging.debug("LDF.reading file")

      from numpy import loadtxt

      if(Precision=="Double"):
          numpyarray=loadtxt(tablefile[0],usecols=(1,2),dtype='f8')  
      elif(Precision=="Simple"):
          numpyarray=loadtxt(tablefile[0],usecols=(1,2),dtype='f4')
      else:
        logging.error("LDF.unrecognized precison, please state either Double or Simple")
        return -1

      if(deletefile==True):
        cmd="rm "+tablefile[0]
        os.system(cmd)
        logging.debug("LDF.Table deleted successfully")

      return numpyarray
    else:
      logging.error("LDF.The requested table was not found and could not be regenerated. Sorry")
      return -1






#this gets the effective refraction index from poitn R0 to xant,yant,zant (default to the core position), all in meters, but kr in 1/km
#ns is the REFRACIVITY AR SEA LEVEL, kr is the scale height in km.
import numpy as np
#NOTE: Use the GetZHSEffectiveactionIndex below to get exactly the same results than in zhaires
# ns= refractivity at sea levele
# kr= scale height

def GetEffectiveRefractionIndex(x0,y0,z0,ns=325,kr=-0.128,zant=0,xant=0,yant=0,stepsize = 1000): #NOTE THAT THE ORDER OF ANTENNA POSITIONS IS ZANT,XANT,YANT (historic compatibility reasons)
        rearth=6370949.0
        R02=x0*x0+y0*y0  #notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada
        h0=(np.sqrt((z0+rearth)*(z0+rearth) + R02 ) - rearth)/1.E3 #altitude of emission, in km

        rh0 = ns*np.exp(kr*h0) #refractivity at emission
        n_h0=1.E0+1.E-6*rh0 #n at emission
#        print("n_h0",n_h0,ns,kr,x0,y0,z0,h0,rh0)

        hd=(zant)/1.E3 #detector altitude

#       Vector from detector to average point on track. Making the integral in this way better guaranties the continuity
#       since the choping of the path will be always the same as you go farther away. If you start at your starting point, for a given geometry,
#       the choping points change with each starting position.

        ux = x0-xant
        uy = y0-yant         #the antenna position, considered to be at the core
        uz = z0-zant

        Rd=np.sqrt(ux*ux + uy*uy)
        kx=ux/Rd
        ky=uy/Rd #!k is a vector from the antenna to the track, that when multiplied by Rd will end in the track and sumed to antenna position will be equal to the track positon
        kz=uz/Rd

#       integral starts at ground
        nint=0
        sum=0.E0

        currpx=0+xant
        currpy=0+yant    #!current point (1st antenna position)
        currpz=zant
        currh=hd
        
        print(Rd)

        while(Rd > stepsize): #if distance projected on the xy plane is more than 10km
          nint=nint+1
          nextpx=currpx+kx*stepsize
          nextpy=currpy+ky*stepsize           #this is the "next" point
          nextpz=currpz+kz*stepsize

          nextR2=nextpx*nextpx + nextpy*nextpy #!se usa el cuadrado, se puede ahorrar la raiz cuadrada
          nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1.E3

          if(np.absolute(nexth-currh) > 1.E-10  ):   #check that we are not going at constant height, if so, the refraction index is constant
              sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
          else:
              sum=sum+np.exp(kr*currh)

          currpx=nextpx
          currpy=nextpy
          currpz=nextpz  #Set new "current" point
          currh=nexth

          Rd=Rd-stepsize #reduce the remaining lenght
        #enddo

        #when we arrive here, we know that we are left with the last part of the integral, the one closer to the track (and maybe the only one)

        nexth=h0

        if(np.absolute(nexth-currh) > 1.E-10 ): #check that we are not going at constant height, if so, the refraction index is constant
          sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
        else:
          sum=sum+np.exp(kr*currh)

        nint=nint+1
        avn=ns*sum/nint
        n_eff=1.E0+1.E-6*avn #average (effective) n
        return n_eff

#This is exactly what is computed in ZHAires
def GetZHSEffectiveRefractionIndex(x0,y0,z0,xant=0,yant=0,zant=0,ns=325,kr=-0.1218,stepsize = 20000):
#      rearth=6371007.0 #new aires
      rearth=6370949.0 #19.4.0
#     Variable n integral calculation ///////////////////
      R02=x0*x0+y0*y0  #!notar que se usa R02, se puede ahorrar el producto y la raiz cuadrada (entro con injz-zXmax -> injz-z0=zXmax
      h0=(np.sqrt( (z0+rearth)*(z0+rearth) + R02 ) - rearth)/1E3    #!altitude of emission

      rh0=ns*np.exp(kr*h0) #!refractivity at emission (this
      n_h0=1+1E-6*rh0 #!n at emission
#        write(*,*) "n_h0",n_h0,ns,kr,x0,y0,injz-z0,h0,rh0
      modr=np.sqrt(R02)

#      print("ZHS n_h0",n_h0,ns,kr,x0,y0,z0,h0,rh0)
        
      if(modr > 1000): #! if inclined shower and point more than 20km from core. Using the core as reference distance is dangerous, its invalid in upgoing showers

#         Vector from average point of track to observer.
          ux = xant-x0
          uy = yant-y0
          uz = zant-z0

#         divided in nint pieces shorter than 10km
          nint=int((modr/stepsize)+1)
          kx=ux/nint
          ky=uy/nint       #k is vector from one point to the next
          kz=uz/nint
#
          currpx=x0
          currpy=y0        #current point (1st is emission point)
          currpz=z0
          currh=h0
#
          sum=0
          for iii in range(0,nint):
            nextpx=currpx+kx
            nextpy=currpy+ky #!this is the "next" point
            nextpz=currpz+kz
            nextR2=nextpx*nextpx + nextpy*nextpy
            nexth=(np.sqrt((nextpz+rearth)*(nextpz+rearth) + nextR2) - rearth)/1E3
#c
            if(np.abs(nexth-currh) > 1E-10  ):
              sum=sum+(np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
              sum=sum+np.exp(kr*currh)
#            endif
#c
            currpx=nextpx
            currpy=nextpy
            currpz=nextpz  #!Set new "current" point
            currh=nexth
#c
          avn=ns*sum/nint
#          print*,"avn:",avn
          n_eff=1+1E-6*avn #!average (effective) n
#c
      else:
#c         withouth integral
          hd=zant/1E3 #!detector altitude(flat earth!)
#
          if(np.abs(hd-h0) > 1E-10):
            avn=(ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
#            print*,"avn2:",avn
          else:
            avn=ns*np.exp(kr*h0)
#           print*,"avn3:",avn
#            print *,"Effective n: h0=hd"
#          endif
          n_eff=1+1E-6*avn #!average (effective) n
#        endif
#c     ///////////////////////////////////////////////////
      return n_eff
#      end

###########################################################################################################################################################################################
def GetRefractionIndexFromDensityFunction(altitude,densityfunction,glasgowconstant):

   return 1+1E-6*densityfunction(altitude)*glasgowconstant*1000 #glasgow constant  is input in m3/kg, but density in g/cm3

#only glasgow implemented for now
def GetSmartEffectiveRefractionIndexFromDensityFunction(r1,r2,densityfunction,depthfunction,glasgowconstant,groundz):
#as implemented on ZHAireS 1.1 (but you also need the same depth and density function than in zhaires, wich is NOT exactly what you get by interpolating the table)
#  
#     Evaluating average density (in g/m.cm2) along a
#     straight line that goes from point r1 to point r2.
#
#     The matter path is defined via
#
#                           point r2
#          xpath/dr1r2 = INTEGRAL'  rho(zv) dl / dr1r2
#                           point r1
#
#     where rho(zv) is the density of the medium, which is supposed to
#     depend only on zv, the vertical altitude. The prime in the
#     integral indicates that it is done along the specified axis.
#
#     Then, we multiply by the glasgowconstant, and we have the refractivity
#
#     Written by: S. J. Sciutto, La Plata 2020., M.J. Tueros, Paris 2020
#
#
#     Arguments:
#     =========
#
#     r1, r2.......... (input, double precision, array(3)) Coordinates
#                      of the initial point.
#     densityfunction  (input, function) a function returning the density (g/cm3) at a given altitude (m)
#     depthpunction    (input, function) a function returning the vertical depth from the top of the atmosphere (g/cm2) at a given altitude (m)
#     groundz          (input, double precision) ground altitude (m). Used to check
#
#     Output:
#     ======
#     rho12........... (output, double precision)  the average index of refraction.
#
      rho12 = 0
      dr1r2 = 0
#
#     Straight line parameterization.
#
      rearth = 6370949.0
      dr1r2 = 0.0
      r1e=r1
      uxp=r2-r1
      dr1r2  = np.linalg.norm(uxp)
#
      if dr1r2 <= 0:
       print("starting distance is 0,this is worrisome")
       dr1r2=0.0
       return
#
      r1e[2]  = r1e[2] + rearth
#
#     Determine number of steps
#      
      nint=int((dr1r2/20000)+1)      
      ds=1.0/nint
#
#     Get altitude of starting point
#      
      zv1=np.linalg.norm(r1e)-rearth
#      
      if zv1<groundz:
        print("starting point is underground, this is wrong")        
        return -1       
#
#     get the depth at the starting point        
#
      x1=depthfunction(zv1)
#      print("starting depth",x1,zv1)               
#
#     now the integral
#         
      suma=0.0      
      for i in np.arange(1,nint+1):     
         s=i*ds

#        get altitude of next point
          
         zv2 = np.linalg.norm(r1e + uxp*s) - rearth
#
#         print("ending depth",zv2) 
         if zv2 < groundz:
           print("point is underground, shadow?",zv2)           
           return -1        
#          
         if np.abs(zv1-zv2) > 0.1:
           x2=depthfunction(zv2)
           suma=suma+(x1-x2)/((zv2-zv1)*100) #put it in cm to get density in g/cm3
#           print("vertical",suma,i,nint,zv1,zv2,(x1-x2)/(zv2-zv1))
#           print("        ",(x1-x2),(zv2-zv1),zv1,zv2,x1,x2)               
                 
         else:
           x2=x1
           suma=suma+densityfunction(zv2)
#           print("horizont",suma,i,nint,zv1,zv2,densityfunction(zv2))             
#
#        now switch x1 with x2
         x1=x2
         zv1=zv2 
#
      rho12=suma/nint
#      print("sum0",suma,nint,dr1r2,rho12)
#
      return 1+1e-6*rho12*glasgowconstant*1000 #glasgow constant  is input in m3/kg, but density in g/cm3


def ReadAiresSry(sry_file,outmode="N/A"):

  zen=GetZenithAngleFromSry(sry_file,outmode)
  azim=GetAzimuthAngleFromSry(sry_file,outmode)
  energy=GetEnergyFromSry(sry_file,outmode)
  primary=GetPrimaryFromSry(sry_file,outmode)
  xmax=GetSlantXmaxFromSry(sry_file,outmode)
  kmxmax,distance,x,y,z=GetKmXmaxFromSry(sry_file,outmode)
  taskname=GetTaskNameFromSry(sry_file,outmode)
  return zen,azim,energy,primary,xmax,distance,taskname

def ReadAiresLgf(lgf_file,outmode="N/A"):

  zen=GetZenithAngleFromSry(lgf_file,outmode)
  azim=GetAzimuthAngleFromSry(lgf_file,outmode)
  energy=GetEnergyFromSry(lgf_file,outmode)
  primary=GetPrimaryFromSry(lgf_file,outmode)
  taskname=GetTaskNameFromSry(lgf_file,outmode)
  return zen,azim,energy,primary,-1,-1,taskname


def GetStatusFromStatus(status_file):
  try:
    datafile=open(status_file,'r')
    with open(status_file, "r") as datafile:
      for line in datafile:
        if 'Aires_Msg' in line:
          line = line.lstrip()
          stripedline=line.split('=',-1)
          status=stripedline[len(stripedline)-1]
          status=status.replace('\'','')
          status=status.replace('\n','')
          #logging.debug("Found status " + status)
          return status
      try:
        status
      except NameError:
        logging.error('warning status not found in status file')
        return 'not found'
  except:
    logging.error("GetStatusFromStatus:file not found or invalid:"+status_file)
    return -1

def GetTmpFromDirs(dirs_file):
  try:
    datafile=open(dirs_file,'r')
    with open(dirs_file, "r") as datafile:
      for line in datafile:
        if 'Aires_DRandomfn' in line:
          line = line.lstrip()
          stripedline=line.split('=',-1)
          tmp=stripedline[len(stripedline)-1]
          tmp=tmp.replace('\'','')
          tmp=tmp.replace('\n','')
          #logging.debug("Found Tmp " + tmp)
          return tmp
      try:
        tmp
      except NameError:
        logging.error('warning Aires_DRandomfn not found in dirs file')
        return 'not found'
  except:
    logging.error("GetTmpFromSDirs:file not found or invalid:"+dirs_file)
    return -1

def DeprecatedReadAiresSry(sry_file,outmode="GRAND"):

    try:
     datafile=open(sry_file,'r')

    except:
      logging.error("ReasAiresSry:file not found or invalid:"+sry_file)
      raise
      return -1, -1, -1, -1, -1

    for line in datafile:
        ## print(line) #debug level 2

        if 'Primary zenith angle:' in line:
            line = line.lstrip()
            stripedline=line.split(' ',-1)
            zen=float(stripedline[len(stripedline)-2])
            if outmode == 'GRAND':
                zen = 180-zen  #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
            logging.debug('Found Zenith ' + str(zen))

        if 'Primary azimuth angle:' in line:
            line = line.lstrip()
            stripedline=line.split(' ',-1)
            azim = float(stripedline[len(stripedline)-2])

            if outmode == 'GRAND':
                azim=azim+180 #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
                if azim>=360:
                    azim= azim-360
            logging.debug('Found Azimuth ' + str(azim))

        if 'Primary energy:' in line:
            line = line.lstrip()
            stripedline=line.split(' ',-1)
            try:
              energy
            except NameError:
              energy = float(stripedline[len(stripedline)-2])
              unit= str(stripedline[len(stripedline)-1])

              if outmode == 'GRAND':
                if unit == "eV\n":
                    energy = energy *1e-18
                if unit == "KeV\n":
                    energy = energy *1e-15
                if unit == "MeV\n":
                    energy = energy *1e-12
                if unit == "GeV\n":
                    energy = energy *1e-9
                if unit == "TeV\n":
                    energy = energy *1e-6
                if unit == "PeV\n":
                    energy = energy *1e-3
                if unit == "EeV\n":
                    energy = energy

              if outmode == 'AIRES':
                if unit == "eV\n":
                    energy = energy *1e-9
                if unit == "KeV\n":
                    energy = energy *1e-6
                if unit == "MeV\n":
                    energy = energy *1e-3
                if unit == "GeV\n":
                    energy = energy
                if unit == "TeV\n":
                    energy = energy *1e3
                if unit == "PeV\n":
                    energy = energy *1e6
                if unit == "EeV\n":
                    energy = energy *1e9

              logging.debug('Found Energy ' + str(energy)) #debug level 1

        if 'Primary particle:' in line:
            line = line.lstrip()
            stripedline=line.split(' ',-1)
            if(len(stripedline)==3):
              primarytype = str(stripedline[len(stripedline)-1])
            elif(len(stripedline)==5):
              primarytype = str(stripedline[len(stripedline)-3])
            elif(len(stripedline)==6):
              primarytype = str(stripedline[len(stripedline)-4])
            elif(len(stripedline)==7):
              primarytype = str(stripedline[len(stripedline)-5])
            else:
              primarytype = "unknown"

            logging.debug('Found Primary ' + primarytype) #debug level 1


        if 'Sl. depth of max. (g/cm2)' in line:
            line = line.lstrip()
            stripedline=line.split(' ',-1)
            xmax = float(stripedline[len(stripedline)-1])
            logging.debug('Found Xmax ' + str(xmax)) #debug level 1

    try:
        zen
    except NameError:
        zen = 0 #If no zenith angle was included in the input file, AIRES defaults to 0
        if outmode == 'GRAND':
          zen = 180-0 #that translates to 180 in GRAND
    try:
        azim
    except NameError:
        azim = 0 #If no azimuth angle was included in the input file, AIRES defaults to 0
        if outmode == 'GRAND':
          azim = 0+180 # that translates to 180
    try:
        energy
    except NameError:
        logging.error('warning energy not found, Aires has no default value,  cannot continue')
        exit()
    try:
        primarytype
    except NameError:
        logging.error('warning primary not found, Aires has no default value, cannot continue')
        exit()

    try:
        xmax
    except NameError:
        logging.info('warning xmax not found')
        xmax=-1

    return zen,azim,energy,primarytype,xmax



if __name__ == '__main__':

    path = sys.argv[1]
    outmode = 'AIRES'
    print(ReadAiresSry(path,outmode))
