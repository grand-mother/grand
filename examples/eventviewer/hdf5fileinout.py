
from astropy.table import Table, Column
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt


################################################################################################################################
#### by M. Tueros. Shared with the Wineware licence. Support and feature requests accepted only accompanied by bottles of wine.
################################################################################################################################
FileFormatVersion=0.0
EventFormatVersion=0.0
hdf5io_compression=True #compressing the file externally still gains 50%!
#this library provides a user interface and defines HDF5 file format for the GRAND experiment
#
# FILE FORMAT (0.0)
#============================
# RunInfo (AstropyTable).
#============================
#This table holds the list of events present on the file, and some general parameters to aid event selection.
#each event is in a separate folder.
#each table has its meta in TableName._table_column_meta__
#     the meta has the units of each value in the table, and maybe something more)
#
# EVENT FORMAT (0.0)
#============================
# EventInfo (AstropyTable)
#============================
# containing general shower information (zenith, azimuth, energy, xmax, xmax position and the like) (for now the meta of EventInfo for compatibility with anne)
# the meta contains the list of suported tables in the event , and if they are present (True) or not (Fase)
#=============================
# ShowerSimInfo (AstropyTable)
#=============================
# showersimulation: simulation details: thinning paramethers, energy cut, ground altitude, injection height, atmospheric model, hadronic model. number of observing levels, number of lateral bins
#=============================
# SignalSimInfo (AstropyTable)
#=============================
# containing details on the electric field computation: index of refraction model, tmin, tbin (this would be the meta of the antenalist)
#=============================
# AntennaInfo (AstropyTable)
#=============================
# containing the names, positions, slopes and probabbly p2p amplitudes and such of each antenna
#=============================
# (SubFolder) AntenaTraces
#=============================
# This subfolder holds one sub-subfolder per antenna (named after the antenna name).
# each sub-subfolder holds one table per trace.
#=============================
# (AntennaName)   efield  (the electric field produced by the shower)
#                 voltage (the antenna response to said field)
#                 noise   (the ambien noise to be added to the voltage)
#                 filteredvoltage (the antenna response after the electronics)
#                 filterednoise (the ambient noise after the electonics - including electronics noise)
#
#=============================
# (SubFolder) ShowerTables
#==============================
# In this SubFolder we store different tables from the shower simulation
#=======================================
# EGround (AstropyTable)
#=======================================
# Contains the energy distribution of particles arriving at ground level
#=======================================
# ELongitudinalProfile (AstropyTable)
#=======================================
# Contains the longitudinal evolution of the energy content in each particle species
#=======================================
# EdepLongitudinalProfile (AstropyTable)
#=======================================
# Contains the longitudinal evolution of the energy deposit by each particle species
#=======================================
# ElowLongitudinalProfile (AstropyTable)
#=======================================
# Contains the longitudinal evolution of the energy of the discarded low energy particles
#=======================================
# NLateralProfile (AstropyTable)
#=======================================
# Contains the lateral distribution of particles at ground level (in radial bins)
#=======================================
# NLongitudinalProfile (AstropyTable)
#=======================================
# Contains the longitudinal evolution of the of the numebr of particles
#=======================================
# NlowLongitudinalProfile (AstropyTable)
#=======================================
# Contains the longitudinal evolution of the number of dicarded  low energy particles
#
################################################################################################################################################################
# Table Savers
################################################################################################################################################################

#all the error handling might be easier defined in a general routine: SaveAstropyTable(Tablepath/TableName)
def SaveRunInfo(OutFilename,RunInfo):
   #TODO: Handle error when OutFilename already contains RunInfo
   RunInfo.write(OutFilename, path="RunInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveEventInfo(OutFilename,EventInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/EventInfo
   EventInfo.write(OutFilename, path=EventName+"/EventInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveAntennaInfo(OutFilename,AntennaInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/AntennaInfo
   #if overwrite=True, it will overwrite the contennts in AntennaInfo, but not on the file (becouse append is True)9
   AntennaInfo.write(OutFilename, path=EventName+"/AntennaInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

#left here for compatibility with some old code [deprecated]
def SaveAntennaInfo4(OutFilename,AntennaInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/AntennaInfo
   #if overwrite=True, it will overwrite the contennts in AntennaInfo, but not on the file (becouse append is True)
   AntennaInfo.write(OutFilename, path=EventName+"/AntennaInfo4", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveAntennaP2PInfo(OutFilename,AntennaP2PInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/AntennaInfo
   #if overwrite=True, it will overwrite the contennts in AntennaInfo, but not on the file (becouse append is True)
   AntennaP2PInfo.write(OutFilename, path=EventName+"/AntennaP2PInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveShowerSimInfo(OutFilename,ShowerSimInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerSimInfo
   ShowerSimInfo.write(OutFilename, path=EventName+"/ShowerSimInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveSignalSimInfo(OutFilename,SignalSimInfo,EventName):
   #TODO: Handle error when OutFilename already contains EventName/SignalSimInfo
   SignalSimInfo.write(OutFilename, path=EventName+"/SignalSimInfo", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveNLongitudinal(OutFilename,NLongitudinal,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/NLongitudinalProfile
   NLongitudinal.write(OutFilename, path=EventName+"/ShowerTables/NLongitudinalProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveELongitudinal(OutFilename,ELongitudinal,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/ELongitudinalProfile
   ELongitudinal.write(OutFilename, path=EventName+"/ShowerTables/ELongitudinalProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveNlowLongitudinal(OutFilename,NlowLongitudinal,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/NlowLongitudinalProfile
   NlowLongitudinal.write(OutFilename, path=EventName+"/ShowerTables/NlowLongitudinalProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveElowLongitudinal(OutFilename,ElowLongitudinal,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/ElowLongitudinalProfile
   ElowLongitudinal.write(OutFilename, path=EventName+"/ShowerTables/ElowLongitudinalProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveEdepLongitudinal(OutFilename,EdepLongitudinal,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/EdepLongitudinalProfile
   EdepLongitudinal.write(OutFilename, path=EventName+"/ShowerTables/EdepLongitudinalProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveLateralDistribution(OutFilename,NLateral,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/NLateral
   NLateral.write(OutFilename, path=EventName+"/ShowerTables/NLateralProfile", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)

def SaveEnergyDistribution(OutFilename,EGround,EventName):
   #TODO: Handle error when OutFilename already contains EventName/ShowerTables/EGRound
   EGround.write(OutFilename, path=EventName+"/ShowerTables/EGround", format="hdf5", append=True,  compression=hdf5io_compression, serialize_meta=True)


######################################################################################################################################################################
# Table Getters
######################################################################################################################################################################

def GetRunInfo(InputFilename):
   #TODO: Handle error when "RunInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   RunInfo=Table.read(InputFilename, path="RunInfo")
   return RunInfo

def GetEventInfo(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/EventInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   EventInfo=Table.read(InputFilename, path=EventName+"/EventInfo")
   return EventInfo

def GetAntennaInfo(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   AntennaInfo=Table.read(InputFilename, path=EventName+"/AntennaInfo")
   return AntennaInfo

def GetAntennaInfo4(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   AntennaInfo=Table.read(InputFilename, path=EventName+"/AntennaInfo4")
   return AntennaInfo

def GetAntennaP2PInfo(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   AntennaInfo=Table.read(InputFilename, path=EventName+"/AntennaP2PInfo")
   return AntennaInfo


def GetAntennaEfield(InputFilename,EventName,AntennaName,OutputFormat="numpy"):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaTraces" does not exists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName" does not exsists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName/Efield" does not exsists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   EfieldTrace=Table.read(InputFilename, path=str(EventName)+"/AntennaTraces/"+str(AntennaName)+"/efield")
   if(OutputFormat=="numpy"):
     EfieldTrace=np.array([EfieldTrace['Time'], EfieldTrace['Ex'],EfieldTrace['Ey'],EfieldTrace['Ez']]).T
   return EfieldTrace

def GetAntennaVoltage(InputFilename,EventName,AntennaName,OutputFormat="numpy"):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaTraces" does not exists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName" does not exsists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName/Efield" does not exsists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   VoltageTrace=Table.read(InputFilename, path=str(EventName)+"/AntennaTraces/"+str(AntennaName)+"/voltage")
   if(OutputFormat=="numpy"):
     VoltageTrace=np.array([VoltageTrace['Time'], VoltageTrace['Vx'],VoltageTrace['Vy'],VoltageTrace['Vz']]).T
   return VoltageTrace

def GetAntennaFilteredVoltage(InputFilename,EventName,AntennaName,OutputFormat="numpy"):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/AntennaTraces" does not exists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName" does not exsists
   #TODO: Handle error when "EventName/AntennaTraces/AntennaName/Efield" does not exsists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   VoltageTrace=Table.read(InputFilename, path=str(EventName)+"/AntennaTraces/"+str(AntennaName)+"/filteredvoltage")
   if(OutputFormat=="numpy"):
     VoltageTrace=np.array([VoltageTrace['Time'], VoltageTrace['Vx'],VoltageTrace['Vy'],VoltageTrace['Vz']]).T
   return VoltageTrace


def GetShowerSimInfo(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/EventInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   ShowerSimInfo=Table.read(InputFilename, path=EventName+"/ShowerSimInfo")
   return ShowerSimInfo

def GetSignalSimInfo(InputFilename,EventName):
   #TODO: Handle error when "EventName" does not exists
   #TODO: Handle error when "EventName/EventInfo" does not exists
   #TODO: Handle error when "InputFilename" is not a file, or a valid file.
   SignalSimInfo=Table.read(InputFilename, path=EventName+"/SignalSimInfo")
   return SignalSimInfo

#######################################################################################################################################################################
# RunInfo Creator
#######################################################################################################################################################################

def CreateRunInfoMeta(RunName):
    #TODO: Handle errors
    RunInfoMeta={
            "FileFormatVersion":FileFormatVersion,
            "RunName":RunName                        #for checking
            #TODO: decide what else goes here
        }
    return RunInfoMeta

def CreateRunInfo(EventName,Primary,Energy,Zenith,Azimuth,XmaxDistance,SlantXmax,HadronicModel,InjectionAltitude,RunInfoMeta):

    a=Column(data=[EventName],name='EventName')   #EventName, states the name of the Task of the simulation, that usually dictates the file names
    b=Column(data=["N/A"],name='EventID')    #An event might have some ID?
    c=Column(data=[Primary],name='Primary')
    d=Column(data=[Energy],name='Energy',unit=u.EeV)
    e=Column(data=[Zenith],name='Zenith',unit=u.deg)
    f=Column(data=[Azimuth],name='Azimuth',unit=u.deg)
    g=Column(data=[XmaxDistance],name='XmaxDistance',unit=u.m)
    h=Column(data=[SlantXmax],name='SlantXmax',unit=u.g/(u.cm*u.cm))
    i=Column(data=[HadronicModel],name='HadronicModel')
    j=Column(data=[InjectionAltitude],name='InjectionAltitude',unit=u.m)
    #Number of Triggered Efield Antennas?
    #Number of Triggered Voltage Antennas?
    #Number of Triggered Voltage+Filter Antennas?

    RunInfo = Table(data=(a,b,c,d,e,f,g,h,i,j),meta=RunInfoMeta)

    return RunInfo

#######################################################################################################################################################################
# RunInfo Getters
#######################################################################################################################################################################

def GetNumberOfEvents(RunInfo):
   #TODO: Handle errors
    return len(RunInfo)

def GetEventName(RunInfo,EventNumber):
   #TODO: Handle errors
    return RunInfo["EventName"][EventNumber]

def GetEventZenith(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["Zenith"][EventNumber])

def GetEventAzimuth(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["Azimuth"][EventNumber])

def GetEventPrimary(RunInfo,EventNumber):
   #TODO: Handle errors
    return str(RunInfo["Primary"][EventNumber])

def GetEventEnergy(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["Energy"][EventNumber])

def GetXmaxPosition(EventInfo):
   #TODO: Handle errors
    return EventInfo["XmaxPosition"]

def GetEventXmaxDistance(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["XmaxDistance"][EventNumber])

def GetEventSlantXmax(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["SlantXmax"][EventNumber])

def GetEventEnergy(RunInfo,EventNumber):
   #TODO: Handle errors
    return float(RunInfo["Energy"][EventNumber])

def GetEventHadronicModel(RunInfo,EventNumber):
   #TODO: Handle errors
    return str(RunInfo["HadronicModel"][EventNumber])

#######################################################################################################################################################################
# EventInfo Creators
#######################################################################################################################################################################

def CreateEventInfoMeta(RunName,EventNumber,EventInfo,ShowerSimInfo,SignalSimInfo,AntennaInfo,AntennaTraces,NLongitudinal,ELongitudinal,NlowLongitudinal,ElowLongitudinal,EdepLongitudinal,LateralDistribution,EnergyDistribution):

    EventInfoMeta = {
        "RunName":RunName,                         #for cross checking
        "EventNumber":EventNumber,                 #Position in the RunInfo Table for easy access
        "EventFormatVersion":EventFormatVersion,   #File Format of the Event (if it might be different)
        "EventInfo": EventInfo,                    #event includes event info
        "ShowerSimInfo":ShowerSimInfo,             #event includes shower simulation info
        "SignalSimInfo":SignalSimInfo,             #event includes signal simulation info
        "AntennaInfo":AntennaInfo,                 #event includes antenna info
        "AntennaTraces":AntennaTraces,             #event includes the traces of each antenna
        "NLongitudinal":NLongitudinal,             #event includes longitudinal tables of number of particles
        "ELongitudinal":ELongitudinal,             #event includes longitudinal tables of energy
        "NlowLongitudinal":NlowLongitudinal,       #event includes longitudinal tables of number discarded low energy particles
        "ElowLongitudinal":ElowLongitudinal,       #event includes longitudinal tables of energy of discarded low energy particles
        "EdepLongitudinal":EdepLongitudinal,       #event includes longitudinal tables of energy deposit
        "LateralDistribution":LateralDistribution, #event includes tables of the lateral distribution of particles at ground
        "EnergyDistribution":EnergyDistribution    #event includes tables of the enertgy distribution of particles at ground
        #TODO: decide what else goes here
        }
    return EventInfoMeta

def CreateEventInfo(EventName,Primary,Energy,Zenith,Azimuth,XmaxDistance,XmaxPosition,XmaxAltitude,SlantXmax,InjectionAltitude,GroundAltitude,Site,Date,FieldIntensity,FieldInclination,FieldDeclination,AtmosphericModel,EnergyInNeutrinos,EventInfoMeta):

    a1=Column(data=[EventName],name='EventName')   #EventName, states the name of the Task of the simulation, that usually dictates the file names
    b1=Column(data=["N/A"],name='EventID')    #An event might have some ID?
    c1=Column(data=[Primary],name='Primary')
    d1=Column(data=[Energy],name='Energy',unit=u.EeV)
    e1=Column(data=[Zenith],name='Zenith',unit=u.deg)
    f1=Column(data=[Azimuth],name='Azimuth',unit=u.deg)
    g1=Column(data=[XmaxDistance],name='XmaxDistance',unit=u.m)
    h1=Column(data=[XmaxPosition],name='XmaxPosition',unit=u.m)
    i1=Column(data=[XmaxAltitude],name='XmaxAltitude',unit=u.m)
    j1=Column(data=[SlantXmax],name='SlantXmax',unit=u.g/(u.cm*u.cm))
    k1=Column(data=[InjectionAltitude],name='InjectionAltitude',unit=u.m)
    l1=Column(data=[GroundAltitude],name='GroundAltitude',unit=u.m)
    m1=Column(data=[Site],name='Site')
    n1=Column(data=[Date],name='Date')
    o1=Column(data=[FieldIntensity],name='BField',unit=u.uT)
    p1=Column(data=[FieldInclination],name='BFieldIncl',unit=u.deg)
    q1=Column(data=[FieldDeclination],name='BFieldDecl',unit=u.deg)
    r1=Column(data=[AtmosphericModel],name='AtmosphericModel')
    s1=Column(data=["N/A"],name='AtmosphericModelParameters')
    t1=Column(data=[EnergyInNeutrinos],name='EnergyInNeutrinos',unit=u.EeV)

    EventInfo = Table(data=(a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1,o1,p1,q1,r1,s1,t1),meta=EventInfoMeta)

    return EventInfo

#######################################################################################################################################################################
# EventInfo Getters
#######################################################################################################################################################################
#TODO: define magnetic field units and coordinate frame propperly. Current implementation uses ZHAireS conventions, but this needs be standarized.

def GetEventBFieldIncl(EventInfo):
   #TODO: Handle errors
    return float(EventInfo["BFieldIncl"])

def GetEventBFieldDecl(EventInfo):
   #TODO: Handle errors
    return float(EventInfo["BFieldDecl"])

def GetGroundAltitude(EventInfo):
   #TODO: Handle errors
    return float(EventInfo["GroundAltitude"])

#######################################################################################################################################################################
# ShowerSim Creators
#######################################################################################################################################################################

def CreateShowerSimInfoMeta(RunName,EventName,ShowerSimulator):

    ShowerSimInfoMeta = {
        "RunName":RunName,                             #For cross checking
        "EventName":EventName,                         #For cross checking
        "ShowerSimulator": ShowerSimulator             #TODO: decide what goes here
    }

    return ShowerSimInfoMeta

def CreateShowerSimInfo(ShowerSimulator,HadronicModel,RandomSeed,RelativeThinning,WeightFactor,GammaEnergyCut,ElectronEnergyCut,MuonEnergyCut,MesonEnergyCut,NucleonEnergyCut,CPUTime,ShowerSimInfoMeta):

    a2=Column(data=[ShowerSimulator],name='ShowerSimulator')
    b2=Column(data=[HadronicModel],name='HadonicModel')
    c2=Column(data=[RandomSeed],name='RandomSeed')
    d2=Column(data=[RelativeThinning],name='RelativeThining') #energy at wich thining starts, relative to the primary energy.
    e2=Column(data=[WeightFactor],name='WeightFactor')
    f2=Column(data=[GammaEnergyCut],name='GammaEnergyCut',unit=u.MeV)
    g2=Column(data=[ElectronEnergyCut],name='ElectronEnergyCut',unit=u.MeV)
    h2=Column(data=[MuonEnergyCut],name='MuonEnergyCut',unit=u.MeV)
    i2=Column(data=[MesonEnergyCut],name='MesonEnergyCut',unit=u.MeV)
    j2=Column(data=[NucleonEnergyCut],name='NucleonEnergyCut',unit=u.MeV)
    k2=Column(data=[CPUTime],name='CPUTime',unit=u.s)
    l2=Column(data=["N/A"],name='OtherParameters')

    ShowerSimInfo = Table(data=(a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2),meta=ShowerSimInfoMeta)

    return ShowerSimInfo

#######################################################################################################################################################################
# ShowerSim Getters
#######################################################################################################################################################################

def GetCPUTime(ShowerSimInfo):
   #TODO: Handle errors
    return float(ShowerSimInfo["CPUTime"])


#######################################################################################################################################################################
# SignalSimInfo Creators
#######################################################################################################################################################################
def CreateSignalSimInfoMeta(RunName,EventName,FieldSimulator):
    SignalSimInfoMeta = {
        "RunName":RunName,                             #For cross checking
        "EventName":EventName,                         #For cross checking
        "FieldSimulator": FieldSimulator,            #TODO: decide what goes here
    }
    return SignalSimInfoMeta

def CreateSignalSimInfo(FieldSimulator,RefractionIndexModel,RefractionIndexParameters,TimeBinSize,TimeWindowMin,TimeWindowMax,SignalSimInfoMeta):

    a3=Column(data=[FieldSimulator],name='FieldSimulator')
    b3=Column(data=[RefractionIndexModel],name='RefractionIndexModel')
    c3=Column(data=[RefractionIndexParameters],name='RefractionIndexModelParameters')
    d3=Column(data=[TimeBinSize],name='TimeBinSize',unit=u.ns)
    e3=Column(data=[TimeWindowMin],name='TimeWindowMin',unit=u.ns)
    f3=Column(data=[TimeWindowMax],name='TimeWindowMax',unit=u.ns)
    g3=Column(data=["N/A"],name='OtherParameters')
    #Number of Triggered Efield Antennas
    #Number of Triggered Voltage Antennas
    #Number of Triggered FilteredVoltage Antennas

    SignalSimInfo = Table(data=(a3,b3,c3,d3,e3,f3,g3),meta=SignalSimInfoMeta)

    return SignalSimInfo

#######################################################################################################################################################################
# SignalSimInfo Getters
#######################################################################################################################################################################
def GetTimeBinSize(SignalSimInfo):
   #TODO: Handle errors
    return float(SignalSimInfo["TimeBinSize"])

def GetTimeWindowMin(SignalSimInfo):
   #TODO: Handle errors
    return float(SignalSimInfo["TimeWindowMin"])

def GetTimeWindowMax(SignalSimInfo):
   #TODO: Handle errors
    return float(SignalSimInfo["TimeWindowMax"])

####################################################################################################################################################################################
#AntennaInfo Creators
####################################################################################################################################################################################

def CreatAntennaInfoMeta(RunName,EventName,VoltageSimulator="N/A",AntennaModel="N/A",EnvironmentNoiseSimulator="N/A",ElectronicsSimulator="N/A",ElectronicsNoiseSimulator="N/A"):
   #TODO: Handle errors
    AntennaInfoMeta = {
           "RunName":RunName,                                        #For cross checking
           "EventName":EventName,                                    #For cross checking
           "VoltageSimulator": VoltageSimulator,                     #TODO: decide what goes here
           "AntennaModel": AntennaModel,
           "EnvironmentNoiseSimulator": EnvironmentNoiseSimulator,
           "ElectronicsSimulator": ElectronicsSimulator,
           "ElectronicsNoiseSimulator": ElectronicsNoiseSimulator
    }
    return AntennaInfoMeta

def CreateAntennaInfo(IDs, antx, anty, antz, slopeA, slopeB, AntennaInfoMeta, P2Pefield=None,P2Pvoltage=None,P2Pfiltered=None,HilbertPeak=None,HilbertPeakTime=None):
   #TODO: Handle errors
    a4=Column(data=IDs,name='ID')
    b4=Column(data=antx,name='X',unit=u.m) #in core cordinates
    c4=Column(data=anty,name='Y',unit=u.m) #in core cordinates
    d4=Column(data=antz,name='Z',unit=u.m) #in core cordinates
    e4=Column(data=slopeA,name='SlopeA',unit=u.m) #in core cordinates
    f4=Column(data=slopeB,name='SlopeB',unit=u.m) #in core cordinates
    data=[a4,b4,c4,d4,e4,f4]

    #this is left here for now, but all the P2P section was moved to a separate table
    if P2Pefield is not None:
      P2Pefield32=P2Pefield.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pefield32[3,:],name='P2P_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[0,:],name='P2Px_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[1,:],name='P2Py_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[2,:],name='P2Pz_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      AntennaInfoMeta.update(P2Pefield=True)

    if P2Pvoltage is not None:
      P2Pvoltage32=P2Pvoltage.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pvoltage32[3,:],name='P2P_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[0,:],name='P2Px_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[1,:],name='P2Py_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[2,:],name='P2Pz_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      AntennaInfoMeta.update(P2Pvoltage=True)

    if P2Pfiltered is not None:
      P2Pfiltered32=P2Pfiltered.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pfiltered32[3,:],name='P2P_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[0,:],name='P2Px_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[1,:],name='P2Py_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[2,:],name='P2Pz_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      AntennaInfoMeta.update(P2Pfiltered=True)

    if HilbertPeak is not None:
      HilbertPeak32=HilbertPeak.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeak32,name='HilbertPeak') #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeak=True)

    if HilbertPeakTime is not None:
      HilbertPeakTime32=HilbertPeakTime.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakTime32,name='HilbertPeakTime',unit=u.ns) #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakTime=True)


    AstropyTable = Table(data=data,meta=AntennaInfoMeta)
    return AstropyTable

####################################################################################################################################################################################
#AntennaInfo Getters
####################################################################################################################################################################################

def GetNumberOfAntennas(AntennaInfo):
   #TODO: Handle errors
    return len(AntennaInfo)

def GetAntIDFromAntennaInfo(AntennaInfo):
   #TODO: Handle errors
   return AntennaInfo["ID"]

def GetXFromAntennaInfo(AntennaInfo):
   #TODO: Handle errors
   return AntennaInfo["X"]

def GetYFromAntennaInfo(AntennaInfo):
   #TODO: Handle errors
   return AntennaInfo["Y"]

def GetZFromAntennaInfo(AntennaInfo):
   #TODO: Handle errors
   return AntennaInfo["Z"]

def GetSlopesFromTrace(Trace):
   return Trace.meta['slopes']

def GetAntennaID(AntennaInfo,AntennaNumber):
   #TODO: Handle errors
    return AntennaInfo["ID"][AntennaNumber]

def GetAntennaPosition(AntennaInfo,AntennaNumber):
   #TODO: Handle errors
    return (AntennaInfo["X"][AntennaNumber],AntennaInfo["Y"][AntennaNumber],AntennaInfo["Z"][AntennaNumber])

def GetAntennaSlope(AntennaInfo,AntennaNumber):
   #TODO: Handle errors
    return (AntennaInfo["SlopeA"][AntennaNumber] ,AntennaInfo["SlopeB"][AntennaNumber])

def GetAntennaPositions(AntennaInfo):
   #TODO: Handle errors
    return (AntennaInfo["X"][:],AntennaInfo["Y"][:],AntennaInfo["Z"][:])

def GetMetaFromTable(AstropyTable):
   return AstropyTable.meta

def GetAntennaInfoFromEventInfo(EventInfo,nant):
   return EventInfo[nant]

####################################################################################################################################################################################
#AntennaP2PInfo Creators
####################################################################################################################################################################################
def CreateAntennaP2PInfo(IDs, AntennaInfoMeta, P2Pefield=None,P2Pvoltage=None,P2Pfiltered=None,HilbertPeakE=None,HilbertPeakV=None,HilbertPeakFV=None,HilbertPeakTimeE=None,HilbertPeakTimeV=None,HilbertPeakTimeFV=None):
   #TODO: Handle errors
    a4=Column(data=IDs,name='ID')
    data=[a4]

    if P2Pefield is not None:
      P2Pefield32=P2Pefield.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pefield32[3,:],name='P2P_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[0,:],name='P2Px_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[1,:],name='P2Py_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      g4=Column(data=P2Pefield32[2,:],name='P2Pz_efield',unit=u.u*u.V/u.m) #p2p Value of the electric field
      data.append(g4)
      AntennaInfoMeta.update(P2Pefield=True)

    if P2Pvoltage is not None:
      P2Pvoltage32=P2Pvoltage.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pvoltage32[3,:],name='P2P_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[0,:],name='P2Px_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[1,:],name='P2Py_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      g4=Column(data=P2Pvoltage32[2,:],name='P2Pz_voltage',unit=u.u*u.V) #p2p Value of the voltage
      data.append(g4)
      AntennaInfoMeta.update(P2Pvoltage=True)

    if P2Pfiltered is not None:
      P2Pfiltered32=P2Pfiltered.astype('f4') #reduce the data type to float 32

      g4=Column(data=P2Pfiltered32[3,:],name='P2P_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[0,:],name='P2Px_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[1,:],name='P2Py_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      g4=Column(data=P2Pfiltered32[2,:],name='P2Pz_filtered',unit=u.u*u.V) #p2p Value of the filtered voltage
      data.append(g4)
      AntennaInfoMeta.update(P2Pfiltered=True)

    if HilbertPeakE is not None:
      HilbertPeakE32=HilbertPeakE.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakE32,name='HilbertPeakE') #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakE=True)

    if HilbertPeakTimeE is not None:
      HilbertPeakTimeE32=HilbertPeakTimeE.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakTimeE32,name='HilbertPeakTimeE',unit=u.ns) #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakTimeE=True)

    if HilbertPeakV is not None:
      HilbertPeakV32=HilbertPeakV.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakV32,name='HilbertPeakV') #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakV=True)

    if HilbertPeakTimeV is not None:
      HilbertPeakTimeV32=HilbertPeakTimeV.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakTimeV32,name='HilbertPeakTimeV',unit=u.ns) #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakTimeV=True)

    if HilbertPeakFV is not None:
      HilbertPeakFV32=HilbertPeakFV.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakFV32,name='HilbertPeakFV') #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakFV=True)

    if HilbertPeakTimeFV is not None:
      HilbertPeakTimeFV32=HilbertPeakTimeFV.astype('f4') #reduce the data type to float 32
      g4=Column(data=HilbertPeakTimeFV32,name='HilbertPeakTimeFV',unit=u.ns) #
      data.append(g4)
      AntennaInfoMeta.update(HilbertPeakTimeFV=True)


    AstropyTable = Table(data=data,meta=AntennaInfoMeta)
    return AstropyTable

########################################################################################################################
# Create and Save traces
########################################################################################################################

def CreateEfieldTable(efield, EventName, EventNumber, AntennaID, AntennaNumber,FieldSimulator, info={}):
    '''
    Create electric field trace in table with header info (numpy array to astropy table)

    Parameters
    ---------
    efield: numpy array
        electric field trace
    EventName: str
        The Name of the Event, for checking or getting other info on the EventInfo Table
    EventNumber:int
        The number of event in the EventInfo table, for fast access
    AntennaID: str
        The ID of the Antenna, for checking or getting other info from the AntennaInfo Table
    AntennaNumber: int
        The number of antenna in the AntennaInfo table, for fast access
    info: dict
        contains meta info

    Returns
    ---------
    efield_ant: astropy table
        The electric field trace as an astropy table

    '''
    info.update({'FieldSimulator':FieldSimulator,'EventName': EventName, 'EventNumber': EventNumber, 'AntennaID':AntennaID, 'AntennaNumber':AntennaNumber})

    a = Column(data=efield.T[0],unit=u.ns,name='Time',)
    b = Column(data=efield.T[1],unit=u.u*u.V/u.meter,name='Ex')
    c = Column(data=efield.T[2],unit=u.u*u.V/u.meter,name='Ey')
    d = Column(data=efield.T[3],unit=u.u*u.V/u.meter,name='Ez')
    efield_ant = Table(data=(a,b,c,d,), meta=info)

    return efield_ant

def SaveEfieldTable(outputfilename,EventName,antennaID,efield):
   #TODO: HAndle error when "efield" already exists
   #TODO: HAndle error when "outputfilename" is not a file, or a valid file.
   #TODO: Adjust format so that we have the relevant number of significant figures. Maybe float64 is not necesary?. What about using float32 or even float 16?
   efield.write(outputfilename, path=EventName+"/AntennaTraces/"+antennaID+"/efield", format="hdf5", append=True, compression=hdf5io_compression,serialize_meta=True)



def CreateVoltageTable(voltage, EventName, EventNumber, AntennaID, AntennaNumber, VoltageSimulator, info={}):
    '''
    Create voltage trace in table with header info  (numpy array to astropy table)

    Parameters
    ---------
    voltage: numpy array
        voltage trace
    EventName: str
        The Name of the Event, for checking or getting other info on the EventInfo Table
    EventNumber:int
        The number of event in the EventInfo table, for fast access
    AntennaID: str
        The ID of the Antenna, for checking or getting other info from the AntennaInfo Table
    AntennaNumber: int
        The number of antenna in the AntennaInfo table, for fast access
    info: dict
        contains meta info

    Returns
    ---------
    voltage_ant: astropy table
        The voltage trace as an Astropy table

    '''
    info.update({'VoltageSimulator':VoltageSimulator,'EventName': EventName, 'EventNumber': EventNumber, 'AntennaID':AntennaID, 'AntennaNumber':AntennaNumber})

    a = Column(data=voltage.T[0],unit=u.ns,name='Time')
    b = Column(data=voltage.T[1],unit=u.u*u.V,name='Vx')
    c = Column(data=voltage.T[2],unit=u.u*u.V,name='Vy')
    d = Column(data=voltage.T[3],unit=u.u*u.V,name='Vz')
    voltage_ant = Table(data=(a,b,c,d,), meta=info)
    return voltage_ant

def SaveVoltageTable(outputfilename,EventName,antennaID,voltage):
   #TODO: HAndle error when "voltage" already exists
   #TODO: HAndle error when "outputfilename" is not a file, or a valid file.
   voltage.write(outputfilename, path=EventName+"/AntennaTraces/"+antennaID+"/voltage", format="hdf5", append=True,compression=hdf5io_compression,serialize_meta=True)

def SaveFilteredVoltageTable(outputfilename,EventName,antennaID,filteredvoltage):
   #TODO: HAndle error when "voltage" already exists
   #TODO: HAndle error when "outputfilename" is not a file, or a valid file.
   filteredvoltage.write(outputfilename, path=EventName+"/AntennaTraces/"+antennaID+"/filteredvoltage", format="hdf5", append=True,compression=hdf5io_compression,serialize_meta=True)

#######################################################################################################################################################################
# Other Stuff to see how things could be done
#######################################################################################################################################################################

#TODO: Split this in 2 functions: GetP2PFromTrace(Trace) to get p2px,p2py,p2pz and p2ptotal (this is not an hdf5io function)
#      used in GetEventP2P(InputFilename, CurrentEventName) to get them for all the antennas in the event
def get_p2p_hdf5(InputFilename,antennamax='All',antennamin=0,usetrace='efield'):

    #TODO: Handle Errors
    #TODO: Handle Errors when the trace type is not found
    '''
    read in all traces from antennamax to antennamin and output the peak to peak electric field and amplitude

    Parameters:
    InputFilename: str
        HDF5File
    antennamin: int
       starting antenna (starts from 0)
    antennamax: int
       final antenna ('All uses all the antennas)
    usetrace: str
       efield, voltage, filteredvoltage
    Output:
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm

    '''
    CurrentRunInfo=GetRunInfo(InputFilename)
    CurrentEventName=GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
    CurrentAntennaInfo=GetAntennaInfo(InputFilename,CurrentEventName)

    if(antennamax=='All'):
     antennamax=len(CurrentAntennaInfo)-1

    # create an array
    p2p_Ex = np.zeros(1+antennamax-antennamin)
    p2p_Ey = np.zeros(1+antennamax-antennamin)
    p2p_Ez = np.zeros(1+antennamax-antennamin)
    p2p_total = np.zeros(1+antennamax-antennamin)
    p2pE=np.zeros(1+antennamax-antennamin)

    for i in range(antennamin,antennamax+1):
      AntennaID=GetAntennaID(CurrentAntennaInfo,i)
      if(usetrace=='efield'):
        trace=GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      elif(usetrace=='voltage'):
        trace=GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      elif(usetrace=='filteredvoltage'):
        trace=GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      else:
        print('You must specify either efield, voltage or filteredvoltage, bailing out')

      p2p= np.amax(trace,axis=0)-np.amin(trace,axis=0)
      p2p_Ex[i-antennamin]= p2p[1]
      p2p_Ey[i-antennamin]= p2p[2]
      p2p_Ez[i-antennamin]= p2p[3]

      amplitude = np.sqrt(trace[:,1]**2. + trace[:,2]**2. + trace[:,3]**2.) # combined components

      p2p_total[i-antennamin] = max(amplitude)-min(amplitude)

    p2pE = np.stack((p2p_Ex, p2p_Ey, p2p_Ez, p2p_total), axis=0)
    return p2pE


#TODO: Split this in 2 functions: GetHilbertFromTrace(Trace) to get hilbert amplitude and time (this is not an hdf5io function)
#      used in GetEventHilbert(InputFilename, CurrentEventName) to get them for all the antennas in the event
def get_peak_time_hilbert_hdf5(InputFilename, antennamax="All",antennamin=0, usetrace="efield", DISPLAY=False) :
#adapted from Valentin Decoene
    #TODO: Handle Errors
    '''
    read in all traces from antennamax to antennamin and output the peak to peak electric field and amplitude

    Parameters:
    InputFilename: str
        HDF5File
    antennamin: int
       starting antenna (starts from 0)
    antennamax: int
       final antenna ('All uses all the antennas)
    usetrace: str
       efield, voltage, filteredvoltage
    Output:
    peaktime: numpy array with the time of the maximum of the hilbert amplitude of the strongest channel
    peakamplitude: numpy array with the maximum of the hilbert amplitude of the strongest channel
    '''
    #DISPLAY=False
    CurrentRunInfo=GetRunInfo(InputFilename)
    CurrentEventName=GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
    CurrentAntennaInfo=GetAntennaInfo(InputFilename,CurrentEventName)

    if(antennamax=='All'):
      antennamax=len(CurrentAntennaInfo)-1

    peaktime= np.zeros(1+antennamax-antennamin)
    peakamplitude= np.zeros(1+antennamax-antennamin)

    for i in range(antennamin,antennamax+1):
      AntennaID=GetAntennaID(CurrentAntennaInfo,i)
      if(usetrace=='efield'):
        trace=GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      elif(usetrace=='voltage'):
        trace=GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      elif(usetrace=='filteredvoltage'):
        trace=GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      else:
        print('You must specify either efield, voltage or filteredvoltage, bailing out')

      from scipy.signal import hilbert
      #now, this is not doing exactly what i was expecting (the hilbert of each component separately. When i plot it, it seems to be mixing channels up)
      #however, it does get the maximum of the modulus of the signal (but i dont understand whats really going on!)
      hilbert_trace=hilbert(trace[:,1:4])
      hilbert_amp = np.abs(hilbert_trace) 												                     #enveloppe de hilbert x, y, z channels
      peakamplitude[i-antennamin]=max([max(hilbert_amp[:,0]), max(hilbert_amp[:,1]), max(hilbert_amp[:,2])]) #find best peakamp for the 3 channels
      peakamplitudelocation=np.where(hilbert_amp == peakamplitude[i-antennamin])
      #this is to assure that there is a maximum amplitude, at that its unique, and that it could be found
      if(peakamplitude[i-antennamin]!=0.0 and np.shape(peakamplitudelocation)==(2,1)):
        peaktime[i-antennamin]=trace[peakamplitudelocation[0],0]                # get the time of the peak amplitude
      else:
        peaktime[i-antennamin]=-1e20

      #PLOT
      if DISPLAY :
        print(peakamplitude[i-antennamin])
        print('peaktime = ',peaktime[i-antennamin])

        hilbert_trace_x=hilbert(trace[:,1])
        hilbert_amp_x = np.abs(hilbert_trace_x)
        hilbert_trace_y=hilbert(trace[:,2])
        hilbert_amp_y = np.abs(hilbert_trace_y)
        hilbert_trace_z=hilbert(trace[:,3])
        hilbert_amp_z = np.abs(hilbert_trace_z)

        hilbert_amp2=np.zeros(np.shape(trace[:,1:4]))
        hilbert_amp2[:,0]=hilbert_amp_x
        hilbert_amp2[:,1]=hilbert_amp_y
        hilbert_amp2[:,2]=hilbert_amp_z
        #peakamplitude[i-antennamin]=max([max(hilbert_amp2[:,0]), max(hilbert_amp2[:,1]), max(hilbert_amp2[:,2])]) #find best peakamp for the 3 channels
        #peaktime[i-antennamin]=trace[np.where(hilbert_amp2 == peakamplitude[i-antennamin])[0],0]                # get the time of the peak amplitude


        fig1 = plt.figure(1,figsize=(7,5), dpi=100, facecolor='w', edgecolor='k')

        ax1=fig1.add_subplot(221)
        plt.plot(trace[:,0], hilbert_amp[:,0], label = 'Hilbert env channel x')
        plt.plot(trace[:,0], trace[:,1], label = 'channel x')
        plt.plot(trace[:,0], hilbert_amp_x, label = 'Hilbertx env channel x')

        plt.legend(loc = 'best')

        ax1=fig1.add_subplot(222)
        plt.plot(trace[:,0], hilbert_amp[:,1], label = 'Hilbert env channel y')
        plt.plot(trace[:,0], trace[:,2], label = 'channel y')
        plt.plot(trace[:,0], hilbert_amp_y, label = 'Hilbertx env channel y')
        plt.legend(loc = 'best')

        ax1=fig1.add_subplot(223)
        plt.plot(trace[:,0], hilbert_amp[:,2], label = 'Hilbert env channel z')
        plt.plot(trace[:,0], trace[:,3], label = 'channel z')
        plt.plot(trace[:,0], hilbert_amp_z, label = 'Hilbertx env channel z')
        plt.legend(loc = 'best')

        ax1=fig1.add_subplot(224)
        plt.plot(trace[:,0], np.sqrt(trace[:,1]**2 + trace[:,2]**2 + trace[:,3]**2), label = 'modulus signal')
        plt.axvline(peaktime[i-antennamin], color='r', label = 'Timepeak')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (muV) (s)')
        plt.legend(loc = 'best')
        plt.show()

    return peaktime, peakamplitude
