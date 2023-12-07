#This module will try to generate Aires/ZHAireS input files for Particle-Only simulations, in order to estimate Xmax.
# Note that you can run them with ZHAireS no problem, by default ZHAireS is off.
# This is actually preferred, to ensure the random seed will give the exact same shower.

import os
import numpy as np
#
def CreateAiresInputHeaderForGRAND(EventName, Primary, Energy, Zenith, Azimuth, RandomSeed="Auto", OutputFile=None, AngularConvention="PointToSource",PrimaryConvention="PDG", OutMode="a" ):
#CreateAiresInputHeaderForGRAND(EventName, Primary, Zenith, Azimuth, Energy)
#Authors: Matias Tueros
#Debugers:
#Testers:
#Creation Date: 8/7/2022
#
#Note that:
# Energy will be rounded to 5 significant digits, becouse then in the Aires summary they are rounded and the summary is used by other scripts to get the parameters.
# Zenith and azimuth to 2 decimals
#EventName, the name of the task. All files in the run will have that name, and some extension. It is usually also the name of the .inp, but not necessarily
#Primary [PDG]: https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf  Proton, Iron, Gamma, or see Aires Manual
#Zenith  [deg, PointToSource (CR) or IncomingDirection (primary direction) [deg] TODO, implement it, find a cool name
#Azimuth [deg, CR or Neutrino convention, geomagnetic, North is 0, West is 90, South 180, East 270, Negative not allowed. deg]:
#Energy  [GeV]
#RandomSeed A number from (0 to 1). "Automatic" produces a random number. "Aires" leaves the work of setting a random seed to Aires.
#OutPutFile: The output filename and path
#OutMode: "a" for append, "w"? to create a new file and erase the old? TODO:check this
#AngularConvention: "Source" or "Incoming"
#PrimaryConvention: "PDG" or "Aires"
#
#
    if OutputFile==None:
        OutputFile=EventName+".inp"
#
#Let start putting things in Aires Conventions
#
#convert from GRAND PDG coding to Aires (see Aires manual section 2.2.1). I could not find PDG code for Gamma, i set it to be 0.
#TODO: Produce a neat algorithm to go from PDG to Aires convention, instead of this case by case thing
# 
    if(PrimaryConvention=="PDG"):   
        if Primary==1000260560:
            Primary="Iron"      
        elif Primary==2212:
            Primary="Proton"
        elif Primary==2112:
            Primary="Neutron"
        elif Primary==111:
            Primary="Pi0"
        elif Primary==211:
            Primary="Pi+"
        elif Primary==-211:
            Primary="Pi-"
        elif Primary==11:
            Primary="Electron"
        elif Primary==-11:
            Primary="Positron"
        elif Primary==0:
            Primary="Gamma"        
        else:
            print("unrecognized primary particle",Primary)
            return
#
# Energy Checks
#        
#TODO: check quality of input, and that it makes sense. Detect if rounding changed the input       
#
# Anguar Convention Checks
# 
    if AngularConvention=="PointToSource":
        print("Angular convention: PointToSource (CR)")
        #TODO: check quality of input, and that it makes sense. Detect if rounding changed the input
    else:
        print("Unrecognized Angular convention")
        return        
    #
    # Random Seed  Checks
    #  
    if RandomSeed=="Auto":
        import random
        RandomSeed=random.uniform(0, 1)
        print("Generated a Random Seed {0:1.9f}".format(RandomSeed))
    elif RandomSeed=="Aires":
        print("Letting Aires Choose the RandomSeed")                  
    elif float(RandomSeed)<1 and float(RandomSeed)>0  :
        RandomSeed=float(RandomSeed)
    else:
        print("Unrecognized RandomSeed",RandomSeed)
        return
#               
    print ("producing input file header on:"+ OutputFile)
#
    base=os.path.basename(OutputFile)
#
    file= open(OutputFile, OutMode)
    file.write('\n##############################################################################################\n')
    file.write('# Aires simulation header generated with CreateAiresInputHeaderFromGRAND                     #\n')
    file.write('##############################################################################################\n')
    task='TaskName '+str(EventName)+ '\n'
    file.write(task)
    prim='PrimaryParticle '+str(Primary) + '\n'
    file.write(prim)
    file.write('PrimaryEnergy {0:.5} GeV\n'.format( round(float(Energy),5) ) )
    file.write('PrimaryZenAngle {0:.2f} deg\n'.format(round(float(Zenith),5) ) )
    file.write('PrimaryAzimAngle {0:.2f} deg Magnetic\n'.format(round(float(Azimuth),5)))
    if(RandomSeed!="Aires"):
        file.write('RandomSeed {0:1.9f}\n'.format(RandomSeed))
    file.write('##############################################################################################\n')
    file.close()
#
#
def CreateAiresAntennaListInp(AntennaPositions,OutputFile,AntennaNames=None,AntennaSelection=['All']):
#
#  AntennaPositions : numpy array with the antenna positions
#  OutputFile will be where the the output will be directed. If the file exists, it will append it
#  AntennaNames: None will name the antennas A0, A1, A2...etc
#                if a list of string is entered, it will use those names.
#  AntennaSelection: All - uses all the antennas
#                    if an array of indices is entered, only antennas on that index will be used
    file= open(OutputFile, "a")
#
    file.write('\n####################################################################################\n')
    file.write('# Antenna List created with CreateAntennaListInp v0.1                              #\n')
    file.write('####################################################################################\n')
#
    nantennas=len(AntennaPositions[:,1])
    print("Number of antennas found:",nantennas)
#
    if(len(AntennaSelection)==1):
      if(AntennaSelection==["All"] or AntennaSelection==["all"]):
        AntennaSelection=np.arange(0,nantennas)
    else:
      print("AntennaSelection",AntennaSelection)    
#
    if(AntennaNames==None):
      AntennaNames=[]
      for i in range(0,nantennas):
        AntennaNames.append("None")
      for i in AntennaSelection:
        AntennaNames[i]="A"+str(i)
#
    for i in AntennaSelection:
#
      file.write("AddAntenna {0:s} {1:11.2f} {2:11.2f} {3:11.2f}\n".format(AntennaNames[i],AntennaPositions[i,0],AntennaPositions[i,1],AntennaPositions[i,2]))
#
#
    file.write('####################################################################################\n')
    file.write('FresnelTime On\n')
    file.write('ZHAireS On\n')
    file.write('####################################################################################\n')
    file.write('# CreateAntennaListInp Finished                                                    #\n')
    file.write('####################################################################################\n\n')
#
    file.close()
#    
def CreateTimeWindowInp( xmaxdistance, OutputFile, Tmin=-250.0, Tmax=750.0):
    #   
    file= open(OutputFile, "a")
    #
    file.write('######################################################################################\n')
    file.write('# Antenna TimeWindow created with CreateTimeWindowInp v0                             #\n')
    file.write('# Xmax to Antenna Distance:{0:.7f} km\n'.format(xmaxdistance/1000))
    file.write('######################################################################################\n')
    file.write('AntennaTimeMin {0:0.2f} ns\n'.format(Tmin))
    file.write('AntennaTimeMax {0:0.2f} ns\n'.format(Tmax))
    file.write('ExpectedXmaxDist {0:0.2f} m\n'.format(xmaxdistance))
    file.write('######################################################################################\n\n')    
    file.close()
#
#
def ZHAireSInputGeneratorForGRAND(SimParametersFile, EventName, Primary, Energy, Zenith, Azimuth, RandomSeed="Auto", OutputFile=None, AngularConvention="PointToSource", PrimaryConvention="PDG", OutMode="a"):
#
# OutputFile
# Name and path of the OutputFile
#
# SimParametersFile:
# A file that has all the relevant parameters for a valid simulation, except  RandomSeed, EventName, Primary, Energy, Zenith, Azimuth
# this File will be concatenated into the generated fie 
#
# RandomSeed: A real number between in (0,1)
# 
# Parameters Below 
# TODO: Check if the truncations and roundings modify the input this is the case and truncate/round if necessary. The truncate part is the dificult one
# Primary: Particle type (PDG conventions)
# Energy: GeV   - can't have more than 5 significant digits for technical reasons (silly but important ones, if this becomes a problem it can be overcome with some work)
# Zenith: GRAND conventions: Angle of direction of motion wrt vertical  (oposite to CR convention that looks at where particle is coming from) (2 decimal places)
# Azimuth: GRAND conventions, 0 is geomagnetic north with, it augments towards West (so Aires Azimuth is) (2 decimal places)
# ZHAiresMode: On or Off (to have ZHAireS radio routines enabled or not)
#
# TODO: Check if SimParametersFileExists
#
    if OutputFile==None:
        OutputFile=EventName+".inp" 
# 
    CreateAiresInputHeaderForGRAND(EventName, Primary, Energy, Zenith, Azimuth, RandomSeed, OutputFile, AngularConvention, PrimaryConvention, OutMode )
#
#   Apending SimParametersFile
#
    fout= open(OutputFile,"a")
    fout.write('#SimParametersFile Follows ###################################################################\n')
    text="# SimParametersFile from " + SimParametersFile+"\n"
    fout.write(text)
    fout.write('##############################################################################################\n')
    fin = open(SimParametersFile, "r")
    data = fin.read()
    fin.close()
    fout.write(data)
    fout.write('#End of SimParametersFile ####################################################################\n')    
#
#
def GenerateZHAireSArrayInputFromSry(SryFile, ArrayPositionsFile, CorePosition, ArrayName, SimParametersFile, Tmin=-50,Tmax=950, OutMode="a"):
    #SryFile: Aires Sry file from wich we will get the shower parameters, including Xmax position (needs to have been run with ZHAireS)
    #ArrayPositionsFile: A file with the array antenna positions, in grand coordinates (AntennaName, X(NS) ,Y(EW) , Z(Above sea level)
    #CorePosition: The desired core position in the array, in grand coordinatesn; TODO:Important. ForNow, this will use only X,Y of the core position and use GroundAltitude in the Sry as the altitude of the core.
    #ArrayName: The ArrayName will be prepended to the EventName to produce a new event name ArrayName_EventName, to output file ArrayName_EventName.inp (be carefull with lenght!)
    #SimParametersFile: A file with the simulations parameters, including the ZHAireS radio - related parameters but excluding antenna positions (that will be generated by this script)
    #Tmin,Tmax: Control time window, in ns
    # 
    import AiresInfoFunctionsGRANDROOT as AiresInfo
    import numpy as np
    #
    # Get The shower Parmeters from the sry file
    #    
    Zenith=AiresInfo.GetZenithAngleFromSry(SryFile,"Aires")
    Azimuth=AiresInfo.GetAzimuthAngleFromSry(SryFile,"Aires")
    # 
    EventName=AiresInfo.GetTaskNameFromSry(SryFile)
    EventName=ArrayName+"_"+EventName
    #
    Primary=AiresInfo.GetPrimaryFromSry(SryFile)
    Energy=AiresInfo.GetEnergyFromSry(SryFile,"GeV")
    XmaxAltitude,XmaxDistance,Xmaxx,Xmaxy,Xmaxz =AiresInfo.GetKmXmaxFromSry(SryFile) #these outputs are in km
    #
    GroundAltitudeKm=AiresInfo.GetGroundAltitudeFromSry(SryFile)/1000.0
    #
    RandomSeed=AiresInfo.GetRandomSeedFromSry(SryFile)
    #
    OutputFile=EventName+".inp"
    #
    if(Xmaxx==-1.0 and Xmaxy==-1.0 and Xmaxz==-1.0):
      print("warning, could not generate {0}, maximum not found or too low: {1:.2f}".format(OutputFile,XmaxAltitude))
      return
    #
    # Create the input header with said parameters. Angular convention needs to be PoitnToSource, becouse data is taken from Aires.sry
    #       
    CreateAiresInputHeaderForGRAND(EventName, Primary, Energy, Zenith, Azimuth, RandomSeed, OutputFile, AngularConvention="PointToSource", PrimaryConvention="Aires", OutMode=OutMode ) 
    #
    # Load the AntennaARray
    #
    ant_pos=np.loadtxt(ArrayPositionsFile,usecols = (1,2,3))
    # 
    file= open(OutputFile,"a")
    file.write("#############################################################################################\n")
    file.write("#Antenna List Generated from {}\n".format(ArrayPositionsFile))
    file.write('#Core Position: {0:.3f} {1:.3f} {2:.3f}\n'.format(CorePosition[0],CorePosition[1],GroundAltitudeKm*1000.0))
    file.write("#############################################################################################\n")
    file.close()
    #
    # Antenna Array is in GRAND coordinates, but ZHAireS needs it wrt ground altitude to the core position.
    # TODO: ZHAireS, uses a cartesian coordinate system centered on the core position, so earth curvature should be taken into account wrt this system.
    # Antenna positions cannot be underground, but they can be negative. So, ground altitude in the input file should be set to the lowest altitude antenna to avoid problems. This means that antenas will be at some height above ground
    # and that the shower will be simulated a little bit further down that in reality, but that should not be a problem. Also keep in mind that if groundaltitude is bellow the real ground level, there might be an offset in the core position.
    #
    # Also, keep in mind that GRAND coordinates might use diferent earth radius than Aires, that uses a round earth. All those subtelties need to be adressed by topography experts when the time to be more precise comes. 
    #
    ant_pos[:,2]=ant_pos[:,2]-GroundAltitudeKm*1000.0
    #move the antenas to Aires coordinates by shifting antennas positions to be centered on the core (note that we use only (x,y) of the input core position
    ant_pos[:,0:2]=ant_pos[:,0:2]-CorePosition[0:2]
    print("WARNING: Using only x,y of the core postion, setting core altitued to ground altitude:",GroundAltitudeKm*1000)
    #
    #use it to create the AntennaList TODO: Add a smart antenna selection, check for topography shadow, etc
    CreateAiresAntennaListInp(ant_pos,OutputFile,AntennaNames=None)
    #
    #Add the time window
    CreateTimeWindowInp(XmaxDistance*1000.0,OutputFile,Tmin,Tmax)    
    #
    #   Apending SimParametersFile
    #
    fout= open(OutputFile,"a")
    fout.write('# SimParametersFile Follows ##################################################################\n')
    text="# SimParametersFile: " + SimParametersFile+"\n"
    fout.write(text)
    fout.write('##############################################################################################\n')
    fin = open(SimParametersFile, "r")
    data = fin.read()
    fin.close()
    fout.write(data)
    fout.write('#End of SimParametersFile ####################################################################\n')   

def GenerateEventParametersFile(EventName, Primary, Energy, Zenith, Azimuth, CorePosition, ArrayName, OutMode="a", TestedPositions="None"):
#    
#   The idea behind having this file is to make it friendly for other simulation programs, and to avoid putting extra things in the Aires/ZHAireS .inp file
#   For now, the only really needed parameters are ArrayName and CorePosition, but if we implement an antenna selection this would be the place to put that information.
#   Also is the place for other information external to Aires/ZHAireS regarding event generation (i.e. parameters of the core position or the antenna selection models)
#
#   Tested Positions should be a list of (x,y,z) tested before this event was generated. This information can be used later for weighting the event.
    OutputFile=EventName+'.EventParameters' 
    fout= open(OutputFile,"a")
    fout.write('#Event Simulation Parameters##################################################################\n')
    fout.write('EventName: {}\n'.format(EventName))
    fout.write('PrimaryEnergy: {0:.5} GeV\n'.format( round(float(Energy),5) ) )
    fout.write('Zenith: {0:.2f} deg\n'.format(round(float(Zenith),5) ) )
    fout.write('Azimuth: {0:.2f} deg Magnetic\n'.format(round(float(Azimuth),5)))
    fout.write('Core Position: {0:.3f} {1:.3f} {2:.3f}\n'.format(CorePosition[0],CorePosition[1],CorePosition[2]))
    fout.write('ArrayName: {}\n'.format(ArrayName))            
    if(TestedPositions!="None"):
      fout.write('#Core positions tested before generating the event ###########################################\n')
      for a in TestedPositions:
        fout.write(a)  
      
    fout.write('#End of EventParameters####################################################################\n')     
    fout.close()

def GetCorePositionFromParametersFile(inp_file):
  try:
    datafile=open(inp_file,'r')
    with open(inp_file, "r") as datafile:
      for line in datafile:
        if 'Core Position:' in line:
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
        #logging.error('warning core position not found, defaulting to (0,0,0)')
        print('warning core position not found, defaulting to (0,0,0)')
        return (0.0,0.0,0.0)
  except:
    #logging.error("GetCorePositionFromInp:file not found or invalid:"+inp_file)
    print("GetCorePositionFromParametersFile:file not found or invalid:"+inp_file)
    raise
    return -1

def GetArrayNameFromParametersFile(inp_file):
  try:
    datafile=open(inp_file,'r')
    with open(inp_file, "r") as datafile:
      for line in datafile:
        if 'ArrayName:' in line:
          line = line.lstrip()
          line = line.rstrip()
          stripedline=line.split(':',-1)
          ArrayName=stripedline[1].lstrip()
          return ArrayName
      try:
        coreposition
      except NameError:
        #logging.error('warning core position not found, defaulting to (0,0,0)')
        print('warning ArrayName not found')
        return "NOT_FOUND"
  except:
    #logging.error("GetCorePositionFromInp:file not found or invalid:"+inp_file)
    print("GetArrayNameFromParametersFile:file not found or invalid:"+inp_file)
    raise
    return -1







#This is kind of my "User Code" to generate the core positions

#generate a core position in an exagon
# In a regular hexagon, the simplest method is to divide it into three rhombuses.
# That way (a) they have the same area, and (b) you can pick a random point in any one rhombus with
# two random variables from 0 to 1. (thanks Greg Kuperberg at stack overflow!).

#the original routine gives a unitary exagon, then its scaled by hexagon size.

#Note that im setting the core altitude at 0. This means that the core will be at xcore,ycore,ZHAireSGroundAltitude, which might not be coincident to the 
#Real altitude of the ground in that place. To do this right, we should get the altitude from the topography and correct things accordingly.
#Remember that in ZHAireS the core will always be at 0,0,GroudAltitude, and the antenna positions need to be put arround that. And antennas cannot be underground

def randinunithex(hexagon_size):
  from math import sqrt
  from random import randrange, random
  vectors = [(-1.,0),(.5,sqrt(3.)/2.),(.5,-sqrt(3.)/2.)]
  x = randrange(3);
  (v1,v2) = (vectors[x], vectors[(x+1)%3])
  (x,y) = (random(),random())
  #
  v=(x*v1[0]+y*v2[0],x*v1[1]+y*v2[1])
  #
  core=np.zeros(3)
  core[0]=v[0]
  core[1]=v[1]
  core[2]=0
  #scale the core
  core=core*hexagon_size
  print("Core Position:",core)
  return core    
    
if __name__ == '__main__':

  import numpy as np
  import sys
  ZHAireSBinary="ZHAireS" #ZHAireS binary used to run the sims (if it is not in the path, include the path)
  #
  # This interface is just how it could be used to run a shower. You could do it differently, for example first generate a group of inp files for xmax determination, then run that, 
  # then generate the zhaires input, then run that
  # Note that the position of the SimParamtersFile, and the ArrayPositions File, and the ArrayName are hard coded in this example
  # to run, create a subdiectory, move there and do (be sure to have ZHAireS installed and in your path)
  # python3 python3 ZHAireSInputGenerator.py TestShower 2212 2.3456E9 56.78 123.45
  
  print (np.size(sys.argv))
  
  if np.size(sys.argv)!=8:
    print ("Arguments = EventName Primary(PDG) Energy (GeV) Zenith (deg, CR) Azimuth (deg, CR) SimParametersFile ArrayPositionsFile (x,y,z [m]) ")
    print ("\n")
    print ("For example: python3 ZHAireSInputGenerator.py TestEvent 2212 1.2345E9 67.89 0 GRAND.VeryCoarse.Subei.Skeleton.inp layout_datachallenge.dat")
  #
  else:
    import subprocess
    #
    #####################################################################
    # USER CODE TO GENERATE EVENT PARAMETERS
    #    
    #Get the parameters  
    EventName = sys.argv[1]
    Primary = int(sys.argv[2])  
    Energy = float(sys.argv[3]) #in deg
    Zenith = float(sys.argv[4]) #in deg
    Azimuth = float(sys.argv[5]) #in deg
    SimParametersFile = sys.argv[6]
    ArrayPositionsFile = sys.argv[7]
    #
    print("Going to try simulating a shower with parameters")
    print("EventName", EventName)
    print("Primary", Primary)
    print("Energy [GeV]", Energy)
    print("Zenith [Deg]" , Zenith)
    print("Azimuth [Deg]", Azimuth)
    print("SimParametersFile",SimParametersFile)
    print("ArrayPositionsFile",ArrayPositionsFile)
    #SimParametersFile="../GRAND.VeryCoarse.Subei.Skeleton.inp"
    #ArrayPositionsFile="../layout_datachallenge.dat"
    #
    # END USER CODE
    ######################################################################
    #
    #Generate the no radio simulation inp
    ZHAireSInputGeneratorForGRAND(SimParametersFile, EventName, Primary, Energy, Zenith, Azimuth, OutMode="w")
    #Run ZHAires for that
    cmd= ZHAireSBinary + '<' + EventName+'.inp'
    print("about to run",cmd)
    out = subprocess.check_call(cmd,shell=True)
    #Get from the .sry the xmax and the parameters again  
    SryFile=EventName+".sry"
    ###################################################################
    # USER CODE to generate random positions
    #
    #Get a random core position in the exagon of 5km  
    CorePosition=randinunithex(5000.00)
    #The CorePosition, for now until we get fancy with the coordinates, is set at X,Y, GroundAltitude, where GroundAltitude is the one used on the sim.
    #This is also done in GenerateZHAireSArrayInputFromSry, that needs to be modified to circunvent this limitation
    import AiresInfoFunctionsGRANDROOT as AiresInfo
    GroundAltitude=AiresInfo.GetGroundAltitudeFromSry(SryFile)
    CorePosition[2]=GroundAltitude
    ArrayName="TestArray"    
    #
    # End USER CODE
    ####################################################################    
    #Generate the EventName.EventParameters
    GenerateEventParametersFile(EventName, Primary, Energy, Zenith, Azimuth, CorePosition, ArrayName, OutMode="w")
    #Generate the input file for the radio sim  
    GenerateZHAireSArrayInputFromSry(SryFile, ArrayPositionsFile, CorePosition, ArrayName, SimParametersFile,OutMode="w")
    #comment the exit statement to also run tthe radio sim after the input file is generated 
    exit()  
    cmd=ZHAireSBinary + '<' + ArrayName + "_" + EventName+'.inp'
    print("about to run",cmd)    
    out = subprocess.check_call(cmd,shell=True)
    #Finally, we delete the No Radio idf. It contains (or it should if everything went well) the same information as the radio simulation (as showers are identical)
    #You could decide to delete other things. (.dat output) Or compress things (the traces). after the root file is generated
    cmd='rm '+EventName+'.idf'
    print("about to run",cmd) 
    out = subprocess.check_call(cmd,shell=True)
    #Generate the RootFile
    cmd='python3 ZHAireSRawToGRANDROOT.py . minimal 0 1 '+EventName+' '+EventName+".root"
    print("about to run",cmd) 
    out = subprocess.check_call(cmd,shell=True)
    #Delete .dat files    
    cmd="rm "+ArrayName+"_"+EventName+"-total-timefresnel-root.dat"
    print("about to run",cmd) 
    out = subprocess.check_call(cmd,shell=True)    

 
