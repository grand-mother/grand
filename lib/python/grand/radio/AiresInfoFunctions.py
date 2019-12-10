#### TODO: set default = 'GRAND'
#### TODO: add docstrings





#this functions will accept GRAND and AIRES outmode, to give the results in each convention
#it will output the primary zen,azim,energy,primarytype, taken from the .inp file present at input_file_path) (assumed only one .inp file per dir)

#aditional output parameters could be:
#  task name,
#  injection depth,
#  ground level,
#  xmax grams (from the sry)
#  xmax position (from the new version sry)
#  AIRES and ZHAireS version
# any other?

#TO DO: treat correctly the different possible primary types, including RASPASS Multi primary
#TO DO: treat correctly the case where a distribution of primary types or energies or angles is put in the input
#TO DO: Check consistency of the output (energy within a range, angles within a range, etc)
#TO DO: have the .sry reader regenerate the summary using AiresSry, if the file is not foun


#6/2019 Matias Tueros, first attempt at python based on original script by Anne Zilles.
#10/209 Migrated them to make it the official library

import sys
from sys import argv
import os

import glob
import logging

#this function reads he Aires .sry file and tries to extract information from the simulation parameters
#using the .sry file is preferred to using the .inp files because:
#Output is standirized: you dont know what you can find in an .inp file (leading spaces, commented lines, repeated keywords, etc)
#Output is what really happened, not what the user whishe it would happen when he did his crappy .inp file.


#lets break the ReadAiresSry into smaller modules. It has the disadventage of opening, scanning and closing the file each time
#but it adds modularity, and closes the files when it does not use it any more. If we need speed,then  we could input datafile, and make a wraper for opening the file




#############

#===========================================================================================================
#===========================================================================================================
def inputfromtxt(input_file_path):
    # I will move this function as soon as I know to where :)
#===========================================================================================================
    '''
    Get shower parameter from inp file for zhaires simulations
    
    Parameters:
        input_file_path: str
            path of inp file
        
    Returns:
        zen: float
            zenith, deg in GRAND conv.
        azim: float
            azimuth, deg in GRAND conv.
        energy: float
            primary energy in eV
        injh: float
            in meters, 100km for CRs
        primarytype: str
            'proton' or 'iron'
        core: numpy array
            core position in meters
        task: str
            ID of shower
    '''
    
    particule = ['eta','pi+','pi-','pi0','Proton','p','proton','gamma','Gamma','electron','Electron','e-','K+','K-','K0L','K0S','K*+'
    ,'muon+','muon-','Muon+','Muon-','mu+','mu-','tau+','tau-','nu(t)','Positron','positron','e+']


    if os.path.isfile(input_file_path) ==  False:  # File does not exist 
        print('--- ATTENTION: inp-file does not exist')
        #exit()
        
    #datafile = file(input_file_path) # why it is not working...
    datafile = open(input_file_path, 'r') 

    for line in datafile:
        if 'PrimaryZenAngle' in line:
            zen=float(line.split(' ',-1)[1])
            zen = 180-zen  #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
        if 'PrimaryAzimAngle' in line:
            azim = float(line.split(' ',-1)[1])+180 #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
            if azim>=360:
                azim= azim-360
        if 'RASPASSHeight' in line:
            injh = float(line.split(' ',-1)[2])
        if 'PrimaryEnergy' in line:
            energy = float(line.split(' ',-1)[1])
            unit= str(line.split(' ',-1)[2])
            if unit == "eV\n":
                energy = energy
            if unit == "GeV\n":
                energy = energy *1e9
            if unit == "EeV\n":
                energy = energy *1e18
        if 'PrimaryParticle' in line:
            primarytype = str(line.split(' ',-1)[1])
            if primarytype[-1]=='\n':
                primarytype=primarytype[0:-1]
            if primarytype[-1]=='\r':
                primarytype=primarytype[0:-1]
        if 'AddSpecialParticle      RASPASSMulti' in line:
            RASPASSMulti_line = line
        if 'TaskName' in line:
            task = str(line.split(' ',-1)[1])
            if task[-1]=='\n':
                task=task[0:-1]
            if task[-1]=='\r':
                task=task[0:-1]
        if '#Core Position:' in line:
            offset = line.split(' ',-1)
            core = np.array([float(offset[2]), float(offset[3]), float(offset[4])])

    try:
        injh
    except NameError:
        injh = 100000. #Case of a cosmic for which no injection height is defined in the input file and is then set to 100 km by ZHAireS
    try:
        energy
    except NameError:
        print('No primary energy found in the ZHAireS input text file.')
        exit()
    try:
        primarytype
    except NameError:
        primarytype = None
    try:
        task
    except NameError:
        task = None
    try:
        core
    except NameError:
        core = np.array([0.,0.,0.])


    if primarytype=='RASPASSMulti':
        tmp = RASPASSMulti_line.split(' ',-1)
        if tmp[-1][-1]=='\n':
            tmp[-1]=tmp[-1][0:-1]
        prod = [x for x in particule if x in set(tmp)]
        ind_prod = np.array([tmp.index(x) for x in prod],dtype=int)
        Wprod = [float(tmp[ind]) for ind in ind_prod+1]
        primarytype = prod[np.argmax(Wprod)]

    if primarytype=='Proton' or primarytype=='K+' or primarytype=='K-' or primarytype=='K0L' or primarytype=='K0S' or primarytype=='K*+':
        primarytype='proton'
    elif primarytype=='gamma' or primarytype=='Gamma' or primarytype=='Electron':
        primarytype='electron'
    elif primarytype=='pi0' or primarytype=='pi-' or primarytype=='pi+':
        primarytype='pion'
    
    # TODO: add observer level -- antenna height to be corrected
    

    #if task:
        #if core.all:
            #return zen,azim,energy,injh,primarytype,core,task
    #if task:
        #return zen,azim,energy,injh,primarytype,task
    #else:
        #return zen,azim,energy,injh,primarytype
        
    return zen,azim,energy,injh,primarytype,core,task

#===========================================================================================================
def _get_positions_zhaires(path):
    '''
    read in antenna positions from Coreas simulations, wrt to sealevel
    
    Parameters:
    datafile: str
        path to folder of run
    
    Returns:
    positions: numpy array
        x,y,z component of antenna positions in meters
    ID_ant: list
        corresponding antenna ID for identification !- [0,1,2,....]
        
        
    NOTE: units assign to positions and slope assuming meters and degree 
    TODO: Test this function. Anne rewote Matias workaround
    '''

    ID_ant, name_ant, x_pos1, y_pos1,z_pos1  = np.genfromtxt(path) 
    ##workarround by Matias
    #token = open(antposfile[0],'r')
    #linestoken=token.readlines()
    #tokens_column_number = 1
    #ID_ant=[]
    #for x in linestoken:
        #ID_ant.append(x.split()[tokens_column_number])
    #token.close()
    
    ### TODO external input needed -- Tutrle
    slopes = np.zeros((len(positions),2)) #(thisis external to ZHAireS)

    #x_pos1=np.asarray(x_pos1)
    #y_pos1=np.asarray(y_pos1)
    #z_pos1=np.asarray(z_pos1)
    positions=np.stack((x_pos1,y_pos1, z_pos1), axis=-1 )*u.m
    
    return positions, ID_ant, slopes


#############

def GetZenithAngleFromSry(sry_file,outmode="GRAND"):
  try:
    datafile=open(sry_file,'r')
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

def GetPrimaryFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
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


#                              Altitude  Distance     x        y        z
#      Location of max.(Km):     3.612     3.66     0.00     0.57     3.61
def GetKmXmaxFromSry(sry_file,outmode="N/A"): #To do. Handle when Xmax is not found, becouse the fit didnt converge, or becouse this is not ZHAireS, or its latest version
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Location of max.(Km)' in line:
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

def GetTaskNameFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Task Name:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          taskname=stripedline[len(stripedline)-1]
          taskname=taskname.replace('\n','')
          #logging.debug("Found taskname " + taskname)
          return taskname
      try:
        taskname
      except NameError:
        logging.error('warning taskname not found, Aires has no default value, cannot continue')
        exit()
  except:
    logging.error("GetZenithAngleFromSry:file not found or invalid:"+sry_file)
    raise
    return -1

def GetRandomSeedFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
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
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Ground altitude:' in line:
          line = line.lstrip()
          stripedline=line.split(' ',-1)
          groundalt=float(stripedline[len(stripedline)-4])
          unit=stripedline[len(stripedline)-3]
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


def GetTotalCPUTimeFromSry(sry_file,outmode="N/A"):
  try:
    datafile=open(sry_file,'r')
    with open(sry_file, "r") as datafile:
      for line in datafile:
        if 'Total CPU time' in line:
          line = line.lstrip()
          stripedline=line.split(':',-1)
          CPUtime=stripedline[len(stripedline)-1]
          #logging.debug("Found taskname " + taskname)
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
    #main ReadAiresInput
    path = sys.argv[1]
    outmode = 'AIRES'
    #print(ReadAiresInput(path,outmode))
    print(ReadAiresSry(path,outmode))
