#!/usr/bin/python
import sys
import os
import glob
import logging
import numpy as np
#import datetime #to get the unix timestamp
import time #to get the unix timestamp
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift  #to shift the time trance for the trigger simulation


logging.basicConfig(level=logging.INFO)
sys.path.append("../Common")
import AiresInfoFunctionsGRANDROOT as AiresInfo
import ZHAireSCompressEvent as ZC
import EventParametersGenerator as EParGen #the functions i use from this file should be moved to root_trees_raw, so that we dont need an additional new file. It will be common to Coreas and ZhAireS.
import raw_root_trees as RawTrees
logging.getLogger('matplotlib').setLevel(logging.ERROR) #this is to shut-up matplotlib


#Author: Matias Tueros, it was Q2 2023 in Barracas, Buenos Aires, Argentina

#TODO:
#TODO: Add energy longitudinal development tables (on zhaires own tree)   
#TODO ASAP: Get Refractivity Model parameters from the sry (unfortunatelly these are not reported in the sry, this will have to wait to the next version of zhaires hopefully after ICRC 2023)
#Maybe move them to the function call for now, and remove the unused Longitudinal switches?

def ZHAireSRawToRawROOT(InputFolder, OutputFileName="GRANDConvention", RunID="SuitYourself", EventID="LookForIt",TaskName="LookForIt", EventName="UseTaskName",ForcedTPre=800,ForcedTPost=2000, TriggerSim=True): 
    '''
    This routine will read a ZHAireS simulation located in InputFolder and put it in the Desired OutputFileName. 
    
    RunID is the ID of the run the event is going to be associated with. If RunID is "SuitYourself" it will try to divide EventID by 1000 and pick the floor number.
    
    EventID is the ID of the Event is going to be associated with. If EventID is "LookForIt" it will asume that the event ID is the last digits of the .sry file,  after the _ and before the .sry extension
    
    TaskName is what ZHAireS uses to name all the output files. And it generally the same as the "EventName". If you want to change the Name of the event when you store it, you can use the "EventName" optional parameter
    If you dont specify a TaskName, it will look for any .sry file in the directory. There should be only one .sry file on the folder for the script to work correctly
    
    OutputFileName is where you want to save the RawRootFile. If you select "GRANDConvention" it will attempt to apply the GRAND data storage convention.
    
    [site]_[date]_[time]_[run_number]_[mod]_[extra], taking the data from the sry and its file name, asuming it is extra_*.sry,  
    
    ForcedTPre forces the tpre of the trace to be the given number in ns, by adding or 0 or removing bins at the start of the trace when necessary
    ForcedTPost forces the tpost of the trace to be the given number in ns, by adding or 0 or removing bins at the end of the trace when necessary
    
    TriggerSim modifies the time window so that the t0 indicates the position of the maximum of the electric field vector, and this is exactly after the desired TPre (ore -TimeWindowMin) time
    
    Note that TPre is -TimeWindowMin from ZHAireS (so its usually positive)
    
    
    The routine is designed for events simulated with ZHAireS 1.0.30a or later.
    
    It requires to be present in the directory: 
    - 1) A TaskName.sry file, with a "TaskName" inside, that is what ZHAireS uses to name all the output files. If "EventName" is not provided in the function input (maybe you want to override it), it will use TaskName as EventName (recommended).
         Note that Aires will truncate the TaskName if it is too long, and the script will fail. Keep your TaskNames with a reasonable lenght.
         ZHAiresRawToRawROOT will take Energy, Primary, Zenith, Azimuth, etc of the simulation from that file.
         This has the upside that you dont need to keep the idf file (which is some MB) and you dont need to have Aires installed on your system.
         But it has the downside that the values are rounded to 2 decimal places. So if you input Zenith was 78.123 it will be read as 78.12
         
         Since in Aires 19.04.10 there is a python interface that could read idf files, we could get the exact value from there, but for now i will keep it as it is. Just dont produce inputs with more than 2 decimal places!
                  
    - 2) All the a##.trace files produced by ZHAireS with the electric field output (you have to have run your sims with CoREASOutput On)
    - 3) a TaskName.EventParameters file , where the event meta-ZHAireS data is stored (for example the core position used to  generate this particular event, the "ArrayName", the event weight, the event time and nanosecond etc.
    - 4) Optional: the necesary longitudinal tables file. If they dont exist, but the TaskName.idf file is present and  and Aires is installed in the system, AiresInfoFunctions will take care of it.
    - 5) In the input file, antenna names must be of the format A+number of antena, i.e., A1, A2....A100...etc 

    The script output is the shower reference frame: this means a cartesian coordinate system with the shower core at 0,0, GroundAltitude (masl).
    Electric fields are output on the X,Y and Z directions in this coordinate system.

    '''
    #Nice to have Feature: If EventID is decideded to be just a number, get the next correlative EventID (with respect to the last inside the file) if no EventID is specified
	
    #The function will write three main sections: ShowerSim, EfieldSim and MetaInfo. 
    #
    SimShowerInfo=True
    SimEfieldInfo=True 
    SimMetaInfo=True
    
    #We will start by storing the tables Coreas and Zhaires have in common.
    #In the future, i might store additional tables (or other sim info) in a separate tree)
    #this is all false becouse is not implemented yet
    #NLongitudinal=False, ELongitudinal=False, NlowLongitudinal=False, ElowLongitudinal=False, EdepLongitudinal=False, LateralDistribution=False, EnergyDistribution=False):
    NLongitudinal=False
    ELongitudinal=False
    NlowLongitudinal=False
    ElowLongitudinal=False
    EdepLongitudinal=False
    LateralDistribution=False
    EnergyDistribution=False
    #########################################################################################################
    #ZHAIRES Sanity Checks
    #########################################################################################################
    #TODO: Handl when InputFolder Does not exist
    CompressedInput=False
    #Provide the advertised functionality for sry file
    if TaskName=="LookForIt":
      sryfile=glob.glob(InputFolder+"/*.sry")

      if(len(sryfile)>1):
        logging.critical("there should be one and only one sry file in the input directory!. cannot continue!")
        return -1

      if(len(sryfile)==0):
        #no sry file found. look to see if the event is compressed. If it is, expand it 
        tgzfile=glob.glob(InputFolder+"/*.tgz")
        if(len(tgzfile)>1):
          logging.critical("no sry file found, at it appears that there are more than one tgz file. cannot continue")
          return -1        
        elif(len(tgzfile)==1): 
          logging.info("no sry file found, but you appear to have one tgz file. attempting to uncompress")
          ZC.ZHAireSCompressEvent(InputFolder,"expand")
          sryfile=glob.glob(InputFolder+"/*.sry")
          if(len(sryfile)==1):
            logging.info("successfully retrieved a sry file from the tgz, lets hope what i uncompressed is the event and not something else")      
            CompressedInput=True
        else:
          logging.critical("there should be one and only one sry file in the input directory!. cannot continue!")
          return -1
    
    else:     
      sryfile=[InputFolder+"/"+TaskName+".sry"]

    #Check Existance of .sry file    
    if  not os.path.isfile(sryfile[0]) :
        logging.critical("Input Summary file not found, {}".format(sryfile))
        return -1
   
    TaskName=AiresInfo.GetTaskNameFromSry(sryfile[0])
    Date=AiresInfo.GetDateFromSry(sryfile[0]).strip()                   #used  
    Site=AiresInfo.GetSiteFromSry(sryfile[0]).strip()      #strip is for removing white spaces and carriage returns before and after             #used
    HadronicModel=AiresInfo.GetHadronicModelFromSry(sryfile[0]).strip()         #Used       
    
    #TODO:idf file is optional in principle, so i dont check for it for now. I should check for the existance of all the required table files and if anyone is missing, check for the existance of the idf file
    idffile=[InputFolder+"/"+EventName+".idf"]

    #provide the advertised functionality for TaskName
    if EventName=="UseTaskName":
      EventName=TaskName

    #provide the advertised funcionality for for EventID
    if EventID=="LookForIt":
      EventID=extract_event_number(sryfile[0])    
    
    #provide the advertised funcionality for for RunID
    if RunID=="SuitYourself":
      RunID=int(EventID)//1000
      
    #provide advertised functionality for OutputFileName  
    if OutputFileName=="GRANDConvention":
      extra=extract_extra(sryfile[0])
      date=convert_date(Date)
      OutputFileName=Site+"_"+date+"_1200_"+str(RunID)+"_"+HadronicModel+"_"+extra+"_"+str(EventID)+".rawroot"
      directory_path="sim_"+Site+"_"+date+"_1200_"+str(RunID)+"_"+HadronicModel+"_"+extra+"_"+str(EventID)
      OutputFileName=directory_path+"/"+OutputFileName
      if not os.path.exists(directory_path):
        # If the directory doesn't exist, create it
        os.makedirs(directory_path)
    
    logging.info("###")
    logging.info("###")
    logging.info("### Starting with event "+EventName+" in "+ InputFolder+" to add to "+OutputFileName +" as Event " + str(EventID) + " of Run " + str(RunID) ) 
	
	###########################################################################################################	
    #Root Sanity Checks 
    ###########################################################################################################
    #TODO: Handle when OutputFileName or RootFileHandle is invalid (does not exist, or is not writable, or whatever we decide is invalid (i.e. must already have the RunInfo trees).)
	#TODO: Handle when RunID is invalid: (not a number, not following conventions if any)
	#TODO: Handle when EventID is invalid: it must be unique at least inside the run, (and probabbly unique among all runs in file, depending on conventions)
    	    
    #############################################################################################################################
    # SimShowerInfo (deals with the details for the simulation that are common between ZHAireS and CoREAS
    #############################################################################################################################
    if(SimShowerInfo):

        #The tree with the Shower information common to ZHAireS and Coreas       
        RawShower = RawTrees.RawShowerTree(OutputFileName)

        #########################################################################################################################
        # Part I: get the information from ZHAIRES (for COREAS, its stuff would be here)
        #########################################################################################################################   
        Primary= AiresInfo.GetPrimaryFromSry(sryfile[0],"GRAND")            #Used
        Zenith = AiresInfo.GetZenithAngleFromSry(sryfile[0],"Aires")        #Used
        Azimuth = AiresInfo.GetAzimuthAngleFromSry(sryfile[0],"Aires")      #Used
        Energy = AiresInfo.GetEnergyFromSry(sryfile[0],"Aires")             #Used
        XmaxAltitude, XmaxDistance, XmaxX, XmaxY, XmaxZ = AiresInfo.GetKmXmaxFromSry(sryfile[0])  #Used all
        #Convert to m
        XmaxAltitude= float(XmaxAltitude)*1000.0
        XmaxDistance= float(XmaxDistance)*1000.0
        XmaxPosition= [float(XmaxX)*1000.0, float(XmaxY)*1000.0, float(XmaxZ)*1000.0]
        SlantXmax=AiresInfo.GetSlantXmaxFromSry(sryfile[0])                 #Used        
        InjectionAltitude=AiresInfo.GetInjectionAltitudeFromSry(sryfile[0]) #Used                         
        
        t1=time.strptime(Date.strip(),"%d/%b/%Y")
        Date = time.strftime("%Y-%m-%d",t1) #adapted to iso
        UnixDate = int(time.mktime(t1))                                     #Used
        FieldIntensity,FieldInclination,FieldDeclination=AiresInfo.GetMagneticFieldFromSry(sryfile[0]) #Used
        AtmosphericModel=AiresInfo.GetAtmosphericModelFromSry(sryfile[0])                              #Used
        LowEnergyModel="Aires-Geisha"		                                                           #Used  #TODO eventualy: Unhardcode This
        EnergyInNeutrinos=AiresInfo.GetEnergyFractionInNeutrinosFromSry(sryfile[0])                    #Used
        EnergyInNeutrinos=EnergyInNeutrinos*Energy #to Convert to GeV
        RandomSeed=AiresInfo.GetRandomSeedFromSry(sryfile[0])                                          #Used
        CPUTime=AiresInfo.GetTotalCPUTimeFromSry(sryfile[0],"N/A")                                     #Used
       
        #These might be "run parameters"
        Lat,Long=AiresInfo.GetLatLongFromSry(sryfile[0])                                               # 
        GroundAltitude=AiresInfo.GetGroundAltitudeFromSry(sryfile[0])                                  #
        GroundDepth=AiresInfo.GetGroundDepthFromSry(sryfile[0])                                        #   
        ShowerSimulator=AiresInfo.GetAiresVersionFromSry(sryfile[0])                                   # 
        ShowerSimulator="Aires "+ShowerSimulator                                                       #
  
        RelativeThinning=AiresInfo.GetThinningRelativeEnergyFromSry(sryfile[0])                        #Used        
        GammaEnergyCut=AiresInfo.GetGammaEnergyCutFromSry(sryfile[0])                                  #Used
        ElectronEnergyCut=AiresInfo.GetElectronEnergyCutFromSry(sryfile[0])                            #Used
        MuonEnergyCut=AiresInfo.GetMuonEnergyCutFromSry(sryfile[0])                                    #Used
        MesonEnergyCut=AiresInfo.GetMesonEnergyCutFromSry(sryfile[0])                                  #Used
        NucleonEnergyCut=AiresInfo.GetNucleonEnergyCutFromSry(sryfile[0])                              #Used

        #These are ZHAireS specific parameters. Other simulators wont have these parameters, and might have others
        WeightFactor=AiresInfo.GetWeightFactorFromSry(sryfile[0])                                      #Used
        EmToHadrFactor=AiresInfo.GetEMtoHadronWFRatioFromSry(sryfile[0])
        MaxWeight=AiresInfo.ComputeMaxWeight(Energy*1E9,RelativeThinning,WeightFactor) 
         
        #Get the atmospheric density profile
        Atmostable=AiresInfo.GetLongitudinalTable(InputFolder,100,Slant=True,Precision="Simple",TaskName=TaskName)   
        Atmosaltitude=np.array(Atmostable.T[0],dtype=np.float32) #its important to make it float32 or it will complain
        Atmosdepth=np.array(Atmostable.T[1],dtype=np.float32)
        Atmosdensity=np.array(Atmostable.T[2],dtype=np.float32)
                                                     
        ############################################################################################################################# 
        # Part II: Fill RawShower TTree	
        ############################################################################################################################

        RawShower.run_number = RunID
        RawShower.sim_name = ShowerSimulator  
        RawShower.event_number = EventID
        RawShower.event_name = EventName
        RawShower.event_date = Date
        RawShower.unix_date = UnixDate
        RawShower.rnd_seed = RandomSeed
        RawShower.energy_in_neutrinos = EnergyInNeutrinos    
        RawShower.energy_primary = [Energy] #TODO: test multiple primaries
        RawShower.azimuth = Azimuth
        RawShower.zenith = Zenith
        RawShower.primary_type = [str(Primary)]  #TODO: test multiple primaries
        RawShower.primary_inj_alt_shc = [InjectionAltitude] #TODO: test multiple primaries
        RawShower.primary_inj_dir_shc=[(-np.sin(np.deg2rad(Zenith))*np.cos(np.deg2rad(Azimuth)),-np.sin(np.deg2rad(Zenith))*np.sin(np.deg2rad(Azimuth)),-np.cos(np.deg2rad(Zenith)))]  #TODO: test multiple primaries

        #using the sine thorem for a triangle with vertices at the earth center, the injection point and the core position (located at groundlitutde)
        rearth=6370949
        logging.info("warning, using round earth with hard coded radius: 6370949m")  #TODO eventualy: Unhardcode This
        sidea=rearth+InjectionAltitude
        sidec=rearth+GroundAltitude
        AngleA=np.deg2rad(180-Zenith)
        AngleC=np.arcsin((sidec/sidea)*np.sin(AngleA))
        AngleB=np.deg2rad(180-np.rad2deg(AngleA)-np.rad2deg(AngleC))
        sideb=sidec*np.sin(AngleB)/np.sin(AngleC)       
        RawShower.primary_inj_point_shc = [(sideb*np.sin(np.deg2rad(Zenith))*np.cos(np.deg2rad(Azimuth)),sideb*np.sin(np.deg2rad(Zenith))*np.sin(np.deg2rad(Azimuth)),sideb*np.cos(np.deg2rad(Zenith)))]  #TODO: test multiple primaries        
        RawShower.site = str(Site) #TODO: Standarize
        RawShower.site_alt=GroundAltitude   #TODO: For now this will do, but we will have a problem when sims take into account round earth influence on zenith, and maybe topography.
        RawShower.site_lat=Lat              #TODO: For now this will do, but we will have a problem when sims take into account very big sites (when the simulation magnetic field depends on core position for example).
        RawShower.site_lon=Long             #TODO: For now this will do, but we will have a problem when sims take into account very big sites.
        
        RawShower.atmos_model = str(AtmosphericModel) #TODO: Standarize
        #TODO:atmos_model_param  # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
        RawShower.atmos_density.append(Atmosdensity)
        RawShower.atmos_depth.append(Atmosdepth)
        RawShower.atmos_altitude.append(Atmosaltitude)

        RawShower.magnetic_field = np.array([FieldInclination,FieldDeclination,FieldIntensity])
        RawShower.xmax_grams = SlantXmax
        RawShower.xmax_pos_shc = XmaxPosition
        RawShower.xmax_distance = XmaxDistance                 
        RawShower.xmax_alt = XmaxAltitude
        RawShower.hadronic_model = HadronicModel
        RawShower.low_energy_model = LowEnergyModel
        RawShower.cpu_time = float(CPUTime) 
        
        #ZHAireS/Coreas
        RawShower.rel_thin = RelativeThinning
        RawShower.maximum_weight = WeightFactor  
        RawShower.hadronic_thinning=1.0
        RawShower.hadronic_thinning_weight=EmToHadrFactor
        #RawShower.rmax= TODO: This is left for the future, when Marty arrives. 
        RawShower.lowe_cut_gamma = GammaEnergyCut
        RawShower.lowe_cut_e = ElectronEnergyCut
        RawShower.lowe_cut_mu = MuonEnergyCut
        RawShower.lowe_cut_meson = MesonEnergyCut
        RawShower.lowe_cut_nucleon = NucleonEnergyCut              

        
        #METAZHAireS (I propose to pass this to a separate tree and section) @TODO: This is repeated in RawMeta, and should be only there.
        EventParametersFile= InputFolder+"/"+TaskName+".EventParameters"
        
        CorePosition=EParGen.GetCorePositionFromParametersFile(EventParametersFile)

        RawShower.shower_core_pos=np.array(CorePosition) # shower core position 

        #Fill the tables
        table=AiresInfo.GetLongitudinalTable(InputFolder,1001,Slant=False,Precision="Simple",TaskName=TaskName)               
        RawShower.long_pd_depth=np.array(table.T[0], dtype=np.float32) 
        RawShower.long_pd_gammas==np.array(table.T[1], dtype=np.float32)

        table=AiresInfo.GetLongitudinalTable(InputFolder,1005,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_slantdepth=np.array(table.T[0], dtype=np.float32)
        RawShower.long_pd_eminus=np.array(table.T[1], dtype=np.float32)

        table=AiresInfo.GetLongitudinalTable(InputFolder,1006,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_pd_eplus=np.array(table.T[1], dtype=np.float32)
        
        table=AiresInfo.GetLongitudinalTable(InputFolder,1008,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_pd_muminus=np.array(table.T[1], dtype=np.float32)

        table=AiresInfo.GetLongitudinalTable(InputFolder,1007,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_pd_muplus=np.array(table.T[1], dtype=np.float32)
        
        table=AiresInfo.GetLongitudinalTable(InputFolder,1291,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_pd_allch=np.array(table.T[1], dtype=np.float32)

        table=AiresInfo.GetLongitudinalTable(InputFolder,1041,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_pd_nuclei=np.array(table.T[1], dtype=np.float32)
        
        ##I will add as hadr pi,K,other cherged, other neutral, proton, antiproton, neutron
        #This means tables: 1211,1213,1091,1092,1021,1022,1023,
        table=AiresInfo.GetLongitudinalTable(InputFolder,1211,Slant=True,Precision="Simple",TaskName=TaskName)                      
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1213,Slant=True,Precision="Simple",TaskName=TaskName)
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1091,Slant=True,Precision="Simple",TaskName=TaskName)
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1092,Slant=True,Precision="Simple",TaskName=TaskName)
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1021,Slant=True,Precision="Simple",TaskName=TaskName)
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1022,Slant=True,Precision="Simple",TaskName=TaskName)
        table+=AiresInfo.GetLongitudinalTable(InputFolder,1023,Slant=True,Precision="Simple",TaskName=TaskName)
        RawShower.long_pd_hadr=np.array(table.T[1], dtype=np.float32)
        
        table=AiresInfo.GetLongitudinalTable(InputFolder,6796,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_ed_depth=np.array(table.T[0], dtype=np.float32) 
        RawShower.long_ed_neutrino=np.array(table.T[1], dtype=np.float32)

        #In order to compute the calorimetric/invisible energy of the cascade we need the energy arriving at ground level.
        #In CORSIKA, this is stored at the last bin of the Cut tables so i add it as an extra line to the table.        
        table=AiresInfo.GetLongitudinalTable(InputFolder,7501,Slant=True,Precision="Simple",TaskName=TaskName)
        ground=AiresInfo.GetLongitudinalTable(InputFolder,5001,Slant=True,Precision="Simple",TaskName=TaskName)
        ground[0]=GroundDepth
        table=np.vstack((table,ground))
                                    
        RawShower.long_ed_gamma_cut=np.array(table.T[1], dtype=np.float32)        

        #In order to compute the calorimetric/invisible energy of the cascade we need the energy arriving at ground level.
        #In CORSIKA, this is stored at the last bin of the Cut tables so i add it as an extra line to the table.  
        table=AiresInfo.GetLongitudinalTable(InputFolder,7705,Slant=True,Precision="Simple",TaskName=TaskName)
        ground=AiresInfo.GetLongitudinalTable(InputFolder,5205,Slant=True,Precision="Simple",TaskName=TaskName)
        ground[0]=GroundDepth
        table=np.vstack((table,ground))                                                          
        RawShower.long_ed_e_cut=np.array(table.T[1], dtype=np.float32)

        #In order to compute the calorimetric/invisible energy of the cascade we need the energy arriving at ground level.
        #In CORSIKA, this is stored at the last bin of the Cut tables so i add it as an extra line to the table.          
        table=AiresInfo.GetLongitudinalTable(InputFolder,7707,Slant=True,Precision="Simple",TaskName=TaskName)                      
        ground=AiresInfo.GetLongitudinalTable(InputFolder,5207,Slant=True,Precision="Simple",TaskName=TaskName)
        ground[0]=GroundDepth
        table=np.vstack((table,ground))                            
        RawShower.long_ed_mu_cut=np.array(table.T[1], dtype=np.float32) 
                
        ##I will add as hadr other cherged, other neutral (becouse aires for energy cut has fewer categories)
        #This means tables: 7591 and 7592
        table=AiresInfo.GetLongitudinalTable(InputFolder,7591,Slant=True,Precision="Simple",TaskName=TaskName)                      
        table+=AiresInfo.GetLongitudinalTable(InputFolder,7592,Slant=True,Precision="Simple",TaskName=TaskName)                              
        #In order to compute the calorimetric/invisible energy of the cascade we need the energy arriving at ground level.
        #In CORSIKA, this is stored at the last bin of the Cut tables so i add it as an extra line to the table.
        #Since longitudinal energy tables are more detailed than for energy cut tables, i need to add several tables more          
        ground=AiresInfo.GetLongitudinalTable(InputFolder,5021,Slant=True,Precision="Simple",TaskName=TaskName)  #neutrons
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5022,Slant=True,Precision="Simple",TaskName=TaskName) #protons
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5023,Slant=True,Precision="Simple",TaskName=TaskName) #pbar       
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5041,Slant=True,Precision="Simple",TaskName=TaskName) #nuclei
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5091,Slant=True,Precision="Simple",TaskName=TaskName) #other charged
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5092,Slant=True,Precision="Simple",TaskName=TaskName) #other neutral
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5211,Slant=True,Precision="Simple",TaskName=TaskName) #pions
        ground+=AiresInfo.GetLongitudinalTable(InputFolder,5213,Slant=True,Precision="Simple",TaskName=TaskName) #kaons
        ground[0]=GroundDepth
        table=np.vstack((table,ground))        
        RawShower.long_ed_hadr_cut=np.array(table.T[1], dtype=np.float32)              
                         
        table=AiresInfo.GetLongitudinalTable(InputFolder,7801,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_ed_gamma_ioniz=np.array(table.T[1], dtype=np.float32)        

        table=AiresInfo.GetLongitudinalTable(InputFolder,7905,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_ed_e_ioniz=np.array(table.T[1], dtype=np.float32) 

        table=AiresInfo.GetLongitudinalTable(InputFolder,7907,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_ed_mu_ioniz=np.array(table.T[1], dtype=np.float32) 

        ##I will add as hadr other cherged, other neutral (becouse aires for energy cut has fewer categories)
        #This means tables: 7891 and 7892        
        table=AiresInfo.GetLongitudinalTable(InputFolder,7891,Slant=True,Precision="Simple",TaskName=TaskName)                      
        table=AiresInfo.GetLongitudinalTable(InputFolder,7892,Slant=True,Precision="Simple",TaskName=TaskName)                      
        RawShower.long_ed_hadr_ioniz=np.array(table.T[1], dtype=np.float32) 
              
        RawShower.fill()
        RawShower.write()


    #############################################################################################################################
    #	SimEfieldInfo
    #############################################################################################################################
    
    #ZHAIRES DEPENDENT
    ending_e = "/a*.trace"
    tracefiles=glob.glob(InputFolder+ending_e)
    #print(tracefiles)
    tracefiles=sorted(tracefiles, key=lambda x:int(x.split(".trace")[0].split("/a")[-1]))
        
    if(SimEfieldInfo and len(tracefiles)>0):

        RawEfield = RawTrees.RawEfieldTree(OutputFileName)                                                                 
        
	    #########################################################################################################################
        # Part I: get the information
        #########################################################################################################################  	
        FieldSimulator=AiresInfo.GetZHAireSVersionFromSry(sryfile[0])                                  #
        FieldSimulator="ZHAireS "+FieldSimulator 
        
        #Getting all the information i need for	RawEfield
        #
        TimeBinSize=AiresInfo.GetTimeBinFromSry(sryfile[0])
        TimeWindowMin=AiresInfo.GetTimeWindowMinFromSry(sryfile[0])
        TimeWindowMax=AiresInfo.GetTimeWindowMaxFromSry(sryfile[0])

        #make an index of refraction table
        #TODO ASAP: Get Refractivity Model parameters from the sry
        RefractionIndexModel="Exponential" #TODO ASPAP: UNHARDCODE THIS
        RefractionIndexParameters=[1.0003250,-0.1218] #TODO ASAP: UNHARDCODE THIS        
        logging.info("Danger!!, hard coded RefractionIndexModel "+str(RefractionIndexModel) + " " + str(RefractionIndexParameters))        
        R0=(RefractionIndexParameters[0]-1.0)*1E6
        #RefrIndex=R0*np.exp(Atmostable.T[0]*RefractionIndexParameters[1]/1000)        
        Atmosrefractivity=R0*np.exp(Atmosaltitude*RefractionIndexParameters[1]/1000.0)

        AntennaN,IDs,antx,anty,antz,antt=AiresInfo.GetAntennaInfoFromSry(sryfile[0])
        ############################################################################################################################# 
        # Fill RawEfield part
        ############################################################################################################################ 

        RawEfield.run_number = RunID
        RawEfield.event_number = EventID
        
        ############################################################################################################################ 
        # Part II.1: Fill Raw Efield per Event values
        ############################################################################################################################ 
        #Populate what we can

        RawEfield.efield_sim=FieldSimulator
              
        RawEfield.refractivity_model = RefractionIndexModel                                       
        RawEfield.refractivity_model_parameters = RefractionIndexParameters                       
         
        RawEfield.atmos_refractivity.append(Atmosrefractivity)                                 
               
        if ForcedTPre!=0:
          RawEfield.t_pre = ForcedTPre
        else:
          RawEfield.t_pre = -TimeWindowMin                                                           
        
        if ForcedTPost!=0:
          RawEfield.t_post = ForcedTPost
        else:
          RawEfield.t_post = TimeWindowMax                                                          
        
        RawEfield.t_bin_size = TimeBinSize                                                              
        
        ############################################################################################################################ 
        # Part II.2: Fill RawEfield per Antenna
        ############################################################################################################################        

        if(IDs[0]==-1 and antx[0]==-1 and anty[0]==-1 and antz[0]==-1 and antt[0]==-1):
	         logging.critical("hey, no antennas found in event sry "+ str(EventID)+" SimEfield not produced")       

        else:		

            #convert to 32 bits so it takes less space 
            antx=np.array(antx, dtype=np.float32)
            anty=np.array(anty, dtype=np.float32)
            antz=np.array(antz, dtype=np.float32)
            antt=np.array(antt, dtype=np.float32)
   
            #TODO: check that the number of trace files found is coincidient with the number of antennas found from the sry  
            logging.info("found "+str(len(tracefiles))+" antenna trace files")


            RawEfield.du_count = len(tracefiles)


            for antfile in tracefiles:
                #print("into antenna", antfile)

                ant_number = int(antfile.split('/')[-1].split('.trace')[0].split('a')[-1]) # index in selected antenna list
                                                                                           # TODO soon: Check for this, and handle what hapens if it fails. Maybe there is a more elegant solution                                                                                                                       
                ant_position=(antx[ant_number],anty[ant_number],antz[ant_number])

                efield = np.loadtxt(antfile,dtype='f4') #we read the electric field as a numpy array

                t_0= antt[ant_number]  
                
                DetectorID = IDs[ant_number]                                                
                
                #print("DetectorID",DetectorID,"AntennaN",AntennaN[ant_number],"ant_number",ant_number,"pos",ant_position,"t0",t_0)

                if(int(AntennaN[ant_number])!=ant_number+1):
                  logging.critical("Warning, check antenna numbers and ids, it seems there is a problem "+str(AntennaN)+" "+str(ant_number+1))
                
                ############################################################################################################################
                # adjust the trace lenght to force the requested tpre and tpost
                ###########################################################################################################################
                #print("before:",np.shape(efield),-TimeWindowMin,TimeWindowMax, (-TimeWindowMin+TimeWindowMax)/RawEfield.t_bin_size)
                
                if ForcedTPre!=0:
                  DeltaTimePre=ForcedTPre+TimeWindowMin
                  DeltaBinsPre=int(np.round(DeltaTimePre/RawEfield.t_bin_size))
                else:
                  DeltaBinsPre=0                    
                  
                if ForcedTPost!=0:
                  DeltaTimePost=ForcedTPost-TimeWindowMax
                  DeltaBinsPost=int(np.round(DeltaTimePost/RawEfield.t_bin_size))
                else:
                  DeltaBinsPost=0  
                               
                if DeltaBinsPre<0 :
                  efield=efield[-DeltaBinsPre:]
                  DeltaBinsPre=0
                  logging.debug("We have to remove "+str(-DeltaBinsPre)+" bins at the start of efield")
                if DeltaBinsPost<0 :
                  efield=efield[:DeltaBinsPost]   
                  logging.debug("We have to remove "+str(-DeltaBinsPost)+" bins at the end of efield")
               
                if DeltaBinsPost>0 or DeltaBinsPre>0:
                  npad = ((DeltaBinsPre, DeltaBinsPost), (0 , 0))
                  efield=np.pad(efield, npad, 'constant')          #TODO. Here I could pad using the "linear_ramp" mode, and pad wit a value that slowly aproached 0.
                  logging.debug("We have to add "+str(DeltaBinsPre)+" bins at the start of efield")
                  logging.debug("We have to add "+str(DeltaBinsPost)+" bins at the end of efield")
                 
                #At this point, the traces are ensued to have a length between  ForcedTpost and a ForcedTpre, and RawEfield.t_pre has the correct time before the nominal t0
               
                #now lets process a "trigger" algorithm that will modify where the trace is located. 
                if(TriggerSim):
                
                  Etotal = np.linalg.norm(efield[:,1:], axis=1) #make the modulus
                  trigger_index = np.argmax(Etotal)                    #look where the maximum happens
                  trigger_time=trigger_index*TimeBinSize
                  
                  if(trigger_time!=RawEfield.t_pre):
                     #plt.plot(Etotal,label="Original")
                     
                     DeltaT=ForcedTPre - trigger_time       
                     ShiftBins=int(DeltaT/TimeBinSize)
                     #print("we need to shift the trace "+ str(DeltaT)+" ns or "+str(ShiftBins)+" Time bins")
                     
                     #this is to assure that, if the maximum is found too late in the trace, we dont move too much outside of the time window (normally, peaks are late in the time window, if you set the time window correctly). 
                     if(ShiftBins < -RawEfield.t_pre):
                       ShiftBins= -RawEfield.t_pre
                       
                     #we could use roll, but roll makes appear the end of the trace at the begining if we roll to much
                     #efield=np.roll(efield,-ShiftBins,axis=0)
                     #Etotal=np.roll(Etotal,-ShiftBins,axis=0)  
                     #so we use scipy shift, that lets you state what value to put for the places you roll
                     
                     efield=shift(efield,(ShiftBins,0),cval=0)
                     Etotal=shift(Etotal,ShiftBins,cval=0)
                 
                     #plt.plot(np.array(efield[:,1]))
                     #plt.plot(np.array(efield[:,2]))
                     #plt.plot(np.array(efield[:,3]))
                     plt.plot(Etotal,label="Shifted")
                     plt.axvline(ForcedTPre/TimeBinSize)
                     plt.legend()
                     plt.show()
                
                ############################################################################################################################# 
                # Part II: Fill RawEfield	 
                ############################################################################################################################ 
                
                #RawEfield.du_id.append(int(AntennaN[ant_number]))
                RawEfield.du_id.append(int(DetectorID[1:])) #TODO:assuming antena names are of the format A+number of antenna! This is a quick fix          
                RawEfield.du_name.append(DetectorID)
                RawEfield.t_0.append(t_0)            

                # Traces
                RawEfield.trace_x.append(np.array(efield[:,1], dtype=np.float32))
                RawEfield.trace_y.append(np.array(efield[:,2], dtype=np.float32))
                RawEfield.trace_z.append(np.array(efield[:,3], dtype=np.float32))

                # Antenna positions in showers's referential in [m]
                RawEfield.du_x.append(ant_position[0])
                RawEfield.du_y.append(ant_position[1])
                RawEfield.du_z.append(ant_position[2])

                         
            #print("Filling RawEfield")
            RawEfield.fill()
            RawEfield.write()
            #print("Wrote RawEfield")

    else:
        logging.critical("no trace files found in "+InputFolder+"Skipping SimEfield") #TODO: handle this exeption more elegantly


    #############################################################################################################################
    #	SimMetaInfo
    #############################################################################################################################

    if(SimMetaInfo):
        #MetaZHAires       		        
        #All this part will go in a function inside of EventParametersGenerator.py, CreateMetaTree(rootfile, eventparameterfile)
        #TODO:Document .EventParemeters file format 
        
        EventParametersFile= InputFolder+"/"+TaskName+".EventParameters"
        
        EParGen.GenerateRawMetaTree(EventParametersFile,RunID,EventID,OutputFileName)
  
    #
    #
    #  FROM HERE ITS LEGACY FROM HDF5 THAT I WILL IMPLEMENT IN A "PROPIETARY" CLASS OF ZHAIRES-ONLY INFORMATION
    #
    #  # The tree with ZHAireS only information
    #  SimZhairesShower = RawTrees.RawZHAireSTree(OutputFileName)
	##############################################################################################################################
	# LONGITUDINAL TABLES (not implemented yet, will need to have ZHAIRES installed on your system and the Official version of AiresInfoFunctions).
	##############################################################################################################################

    if(NLongitudinal):
        #the gammas table
        table=AiresInfo.GetLongitudinalTable(InputFolder,1001,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteSlantDepth(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteNgammas(HDF5handle, RunID, EventID, table.T[1])

        #the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,1205,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteVerticalDepth(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteNeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the e plus (yes, the positrons)
        table=AiresInfo.GetLongitudinalTable(InputFolder,1006,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNeplus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus mu minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1207,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1007,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNmuplus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus pi munus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1211,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNpiplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1011,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNpiplus(HDF5handle, RunID, EventID, table.T[1])

        #and the all charged
        table=AiresInfo.GetLongitudinalTable(InputFolder,1291,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNallcharged(HDF5handle, RunID, EventID, table.T[1])

	##############################################################################################################################
	# Energy LONGITUDINAL TABLES (very important to veryfy the energy balance of the cascade, and to compute the invisible energy)
	##############################################################################################################################
    if(ELongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,1501,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,1705,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus mu minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1707,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus pi minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1711,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEpiplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the k plus k minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1713,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEkplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the neutrons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1521,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEneutrons(HDF5handle, RunID, EventID, table.T[1])

        #the protons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1522,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEprotons(HDF5handle, RunID, EventID, table.T[1])

        #the anti-protons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1523,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEpbar(HDF5handle, RunID, EventID, table.T[1])

        #the nuclei
        table=AiresInfo.GetLongitudinalTable(InputFolder,1541,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEnuclei(HDF5handle, RunID, EventID, table.T[1])

        #the other charged
        table=AiresInfo.GetLongitudinalTable(InputFolder,1591,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEother_charged(HDF5handle, RunID, EventID, table.T[1])

        #the other neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,1592,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEother_neutral(HDF5handle, RunID, EventID, table.T[1])

        #and the all
        table=AiresInfo.GetLongitudinalTable(InputFolder,1793,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEall(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # NLowEnergy Longitudinal development
    #################################################################################################################################
    if(NlowLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7001,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNlowgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7005,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNloweplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7006,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNloweplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7207,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNlowmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7091,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNlowother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7092,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteNlowother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # ELowEnergy Longitudinal development
    #################################################################################################################################
    if(ElowLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7501,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteElowgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7505,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEloweplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7506,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEloweplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7707,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteElowmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7591,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteElowother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7592,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteElowother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # EnergyDeposit Longitudinal development
    #################################################################################################################################
    if(EdepLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7801,Slant=True,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7805,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7806,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepeplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7907,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7891,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7892,Slant=False,Precision="Simple",TaskName=TaskName)
        SimShower.SimShowerWriteEdepother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # Lateral Tables
    #################################################################################################################################
    if(LateralDistribution):
        #the gammas
        table=AiresInfo.GetLateralTable(InputFolder,2001,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFradius(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteLDFgamma(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2205,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFeplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2006,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFeplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2207,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2007,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFmuplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2291,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFallcharged(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # Energy Distribution at ground Tables
    #################################################################################################################################
    if(EnergyDistribution):
        #the gammas
        table=AiresInfo.GetLateralTable(InputFolder,2501,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_energy(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteEnergyDist_gammas(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2705,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_eplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2506,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_eplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2707,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_muplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2507,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_muplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2791,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_allcharged(HDF5handle, RunID, EventID, table.T[1])

    logging.info("### The event written was " + EventName)

    # f.Close()
    # print("****************CLOSED!")
    
    if(CompressedInput==True):
      logging.info("Hopefully deleting the uncompressed files " + EventName)
      ZC.ZHAireSCompressEvent(InputFolder,"delete")
    
    return EventName


def CheckIfEventIDIsUnique(EventID, f):
    # Try to get the tree from the file
    try:
        SimShower_tree = f.rawshower
        # This readout should be done with RDataFrame, but it crashes on evt_id :/
        # So doing it the old, ugly way
        SimShower_tree.Draw("evt_id", "", "goff")
        EventIDs = np.frombuffer(SimShower_tree.GetV1(), dtype=np.float64, count=SimShower_tree.GetSelectedRows()).astype(int)

    # SimShower TTree doesn't exist -> look for SimEfield
    except:
        try:
            SimEfield_tree = f.SimEfield
            SimEfield_tree.Draw("evt_id", "", "goff")
            EventIDs = np.frombuffer(SimEfield_tree.GetV1(), dtype=np.float64, count=SimEfield_tree.GetSelectedRows()).astype(int)

        # No trees - any EventID will do
        except:
            return True

    # If the EventID is already in the trees' EventIDs, return False
    if EventID in EventIDs:
        return False

    return True


def extract_event_number(file_path):
    # Get the base name of the file without the extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Split the file name using underscores
    parts = file_name.split('_')

    # The EventNumber is the last part of the split
    event_number = parts[-1]

    return event_number

def extract_extra(file_path):
    # Get the base name of the file without the extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Split the file name using underscores
    parts = file_name.split('_')

    # The extra is the first part of the split
    extra = parts[0]

    return extra

def convert_date(date_str):
    # Convert input string to a struct_time object
    date_struct = time.strptime(date_str, "%d/%b/%Y")

    # Format the struct_time object as a string in YYYYMMDD format
    formatted_date = time.strftime("%Y%m%d", date_struct)

    return formatted_date
    
    
if __name__ == '__main__':

    if len(sys.argv)==6 :
        InputFolder=sys.argv[1]
        mode=sys.argv[2]
        RunID=int(sys.argv[3])
        try:
         EventID=int(sys.argv[4])
        except:
         EventID=sys.argv[4]  
        
        OutputFileName=sys.argv[5]
	    
    elif len(sys.argv)==2:
        InputFolder=sys.argv[1]
        mode="standard"
        RunID="SuitYourself"
        EventID="LookForIt"
        OutputFileName="GRANDConvention"	
    else:
        print("Please point me to a directory with some ZHAires output, and indicate the mode RunID, EventID and output filename...nothing more, nothing less!")
        print("i.e ZHAiresRawToRawROOT ./MyshowerDir standard RunID EventID MyFile.root")
        print("i.e. python3 ZHAireSRawToRawROOT.py ./GP10_192745211400_SD075V standard 0 3  GP10_192745211400_SD075V.root")
        print("or point me to a directory and i will take care of the rest automatically as i see fit.")
        mode="exit"



    if(mode=="standard"): 
        ZHAireSRawToRawROOT(InputFolder, OutputFileName, RunID, EventID)

	#elif(mode=="full"):

	#	ZHAireSRawToRawROOT(OutputFileName,RunID,EventID, InputFolder, SimEfieldInfo=True, NLongitudinal=True, ELongitudinal=True, NlowLongitudinal=True, ElowLongitudinal=True, EdepLongitudinal=True, LateralDistribution=True, EnergyDistribution=True)

	#elif(mode=="minimal"):
	
	#	ZHAireSRawToRawROOT(OutputFileName,RunID,EventID, InputFolder, SimEfieldInfo=True, NLongitudinal=False, ELongitudinal=False, NlowLongitudinal=False, ElowLongitudinal=False, EdepLongitudinal=False, LateralDistribution=False, EnergyDistribution=False)

    else:

        print("please enter the mode: standard (full or minimal still not implemented")
	
 

