#!/usr/bin/python
## Conversion of Coreas simulations to GRANDRaw ROOT files
## by Jelena Köhler

import sys
import glob
import datetime #to get the unix timestamp
import time #to get the unix timestamp
from CorsikaInfoFuncs import * # this is in the same dir as this file
sys.path.append("../Common")
import raw_root_trees as RawTrees # this is in Common. since we're in CoREASRawRoot, this is in ../Common
#from grand.io.root_trees import * # this is home/grand/grand (at least in docker) or ../../grand  #this is not needded becouse root_trees is imported on RawTrees

def CoreasToRawRoot(path):
  """
  put meaningful comments here - maybe after I clean up the structure of this file

  """
  #WARNING: output file name hardcoded in the root tree section

  print("-----------------------------------------")
  print("------ COREAS to RAWROOT converter ------")
  print("-----------------------------------------")
  ###############################
  # Part A: load Corsika files  #
  ###############################

  print("Checking", path, "for *.reas and *.inp files (shower info).")
  # TODO: check if the IDs in SIM.reas and RUN.inp match
  # TODO: maybe specify RUN number as input value - or as choice when there is more than one 

  # ********** load SIM.reas **********
  # find reas files
  if glob.glob(path + "SIM??????-*.reas"):
      available_reas_files = glob.glob(path + "SIM??????-*.reas") # these are from parallel runs - I will mostly have these
  else:
      available_reas_files = glob.glob(path + "SIM??????.reas") # these are from normal runs


  # reas status messages
  if len(available_reas_files) == 0:
    print("[ERROR] No reas file found in this directory. Please check directory and try again.")
    quit()
  elif len(available_reas_files) > 1:
    print("Found", available_reas_files)
    print("[WARNING] More than one reas file found in directory. Only reas file", available_reas_files[0], "will be used.")
    reas_input = available_reas_files[0]
  else:
    print("Found", available_reas_files)
    reas_input = available_reas_files[0]
    print("Converting reas file", reas_input, "to GRANDroot.")
  print("*****************************************")


  # ********** load RUN.inp **********
  # find inp files
  # inp files can be named with SIM or RUN, so we will search for both
  available_inp_files_sim = glob.glob(path + "SIM??????.inp")
  available_inp_files_run = glob.glob(path + "RUN??????.inp")

  available_inp_files = available_inp_files_sim + available_inp_files_run


  # inp status messages
  if len(available_inp_files) == 0:
    print("[ERROR] No input file found in this directory. Please check directory and try again.")
    quit()
  elif len(available_inp_files) > 1:
    print("Found", available_inp_files)
    print("[WARNING] More than one input file found in directory. Only input file", available_inp_files[0], "will be used.")
    inp_input = available_inp_files[0]
  else:
    print("Found", available_inp_files)
    inp_input = available_inp_files[0]
    print("Converting input file", inp_input, "to GRANDroot.")


  # ********** load traces **********
  print("Checking subdirectories for *.dat files (traces).")
  available_traces = glob.glob(path + "SIM??????_coreas/*.dat")
  print("Found", len(available_traces), "*.dat files (traces).")
  print("*****************************************")
  # in each dat file:
  # time stamp and the north-, west-, and vertical component of the electric field

  # ********** load log file **********
  """
  This is just until I have a chance to change the Coreas output so that the
  reas file includes the first interaction as an output parameter.

  For now, we just want this for the height the of first interaction.
  """
  log_file = glob.glob(path + "*.log")

  if len(log_file) == 0:
    print("[ERROR] No log file found in this directory. Please check directory and try again.")
    quit()
  elif len(log_file) > 1:
    print("Found", available_inp_files)
    print("[WARNING] More than one log file found in directory. Only log file", log_file[0], "will be used.")
    log_file = log_file[0]
  else:
    print("Found", log_file)
    log_file = log_file[0]
    print("Extracting info from log file", log_file, "for GRANDroot.")
  print("*****************************************")

  first_interaction = read_first_interaction(log_file) # height of first interaction

  ###############################
  # Part B: Generate ROOT Trees #
  ###############################

  #########################################################################################################################
  # Part B.I.i: get the information from Coreas input files
  #########################################################################################################################   
  # from reas file
  CoreCoordinateNorth = read_params(reas_input, "CoreCoordinateNorth") * 100 # convert to m
  CoreCoordinateWest = read_params(reas_input, "CoreCoordinateWest") * 100 # convert to m
  CoreCoordinateVertical = read_params(reas_input, "CoreCoordinateVertical") * 100 # convert to m
  CorePosition = [CoreCoordinateNorth, CoreCoordinateWest, CoreCoordinateVertical]

  TimeResolution = read_params(reas_input, "TimeResolution")
  AutomaticTimeBoundaries = read_params(reas_input, "AutomaticTimeBoundaries")
  TimeLowerBoundary = read_params(reas_input, "TimeLowerBoundary")
  TimeUpperBoundary = read_params(reas_input, "TimeUpperBoundary")
  ResolutionReductionScale = read_params(reas_input, "ResolutionReductionScale")

  GroundLevelRefractiveIndex = read_params(reas_input, "GroundLevelRefractiveIndex") # refractive index at 0m asl

  RunID = read_params(reas_input, "RunNumber")
  EventID = read_params(reas_input, "EventNumber")
  GPSSecs = read_params(reas_input, "GPSSecs")
  GPSNanoSecs = read_params(reas_input, "GPSNanoSecs")
  FieldDeclination = read_params(reas_input, "RotationAngleForMagfieldDeclination") # in degrees

  zenith = read_params(reas_input, "ShowerZenithAngle") # TODO: change this! these are just in the long reas files
  azimuth = read_params(reas_input, "ShowerAzimuthAngle")

  Energy = read_params(reas_input, "PrimaryParticleEnergy") # in GeV
  Primary = read_params(reas_input, "PrimaryParticleType") # as defined in CORSIKA -> TODO: change to PDG system
  DepthOfShowerMaximum = read_params(reas_input, "DepthOfShowerMaximum") # slant depth in g/cm^2
  DistanceOfShowerMaximum = read_params(reas_input, "DistanceOfShowerMaximum") * 100 # geometrical distance of shower maximum from core in m
  FieldIntensity = read_params(reas_input, "MagneticFieldStrength") # in Gauss -> TODO: change to mT
  FieldInclination = read_params(reas_input, "MagneticFieldInclinationAngle") # in degrees, >0: in northern hemisphere, <0: in southern hemisphere
  GeomagneticAngle = read_params(reas_input, "GeomagneticAngle") # in degrees


  # from inp file
  nshow = read_params(inp_input, "NSHOW") # number of showers - should always be 1 for coreas, so maybe we dont need this parameter at all
  ectmap = str(read_params(inp_input, "ECTMAP"))
  maxprt = str(read_params(inp_input, "MAXPRT"))
  radnkg = str(read_params(inp_input, "RADNKG"))
  print("*****************************************")

  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # fix these values! for now: placeholders
  # TODO: read and store all seeds as a list
  print("[WARNING] RandomSeed is hardcoded")
  RandomSeed = [1,2,3,4,5,6]
  
  ecuts = [1,2,3,4] 
  print("[WARNING] ecuts is hardcoded")
  # 0: hadrons & nuclei, 1: muons, 2: e-, 3: photons
  GammaEnergyCut    = ecuts[3]
  ElectronEnergyCut = ecuts[2]
  MuonEnergyCut     = ecuts[1]
  HadronEnergyCut   = ecuts[0]
  NucleonEnergyCut  = ecuts[0]
  MesonEnergyCut    = HadronEnergyCut # mesons are hadronic, so this should be fine

  parallel = [1,2] # COREAS-only
  print("[WARNING] parallel is hardcoded")
  ECTCUT = parallel[0]
  ECTMAX = parallel[1]

  # PARALLEL = [ECTCUT, ECTMAX, MPIID, FECTOUT]
  # ECTCUT: limit for subshowers GeV
  # ECTMAX: maximum energy for complete shower GeV
  # MPIID: ID for mpi run (ignore for now)
  # T/F flag for extra output file (ignore for now)

  # elmflg = ["T", "T"] # COREAS-only (ignore for now)

  # In Zhaires converter: RelativeThinning, WeightFactor
  # I have:
  Thin  = [1,2,3] 
  print("[WARNING] THIN is hardcoded")
  # THIN = [limit, weight, Rmax]
  ThinH = [1,2] 
  print("[WARNING] THINH is hardcoded")
  # THINH = [limit, weight] for hadrons

  ##########################################
  # get all info from the long file
  pathLongFile = glob.glob(path + "DAT??????.long")[0]

  # the long file has an annoying setup, which I (very inelegantly) circumvent with this function:
  n_data, dE_data, hillas_parameter = read_long(pathLongFile)
  # TODO: there's still an issue with the hillas_parameter in read_long

  #**** particle distribution
  particle_dist = n_data
  # DEPTH, GAMMAS, POSITRONS, ELECTRONS, MU+, MU-, HADRONS, CHARGED, NUCLEI, CHERENKOV
  pd_depth = particle_dist[:,0]
  pd_gammas = particle_dist[:,1]
  pd_positrons = particle_dist[:,2]
  pd_electrons = particle_dist[:,3]
  pd_muP = particle_dist[:,4]
  pd_muN = particle_dist[:,5]
  pd_hadrons = particle_dist[:,6]
  pd_charged = particle_dist[:,7]
  pd_nuclei = particle_dist[:,8]
  pd_cherenkov = particle_dist[:,9]

  #**** energy deposit
  energy_dep = dE_data
  # the depth here is not the same as for the particle dist, because that would be too easy (they are usually shifted by 5)
  # DEPTH, GAMMA, EM IONIZ, EM CUT, MU IONIZ, MU CUT, HADR IONIZ, HADR CUT, NEUTRINO, SUM
  ed_depth = energy_dep[:,0]
  ed_gamma = energy_dep[:,1]
  ed_em_ioniz = energy_dep[:,2]
  ed_em_cut = energy_dep[:,3]
  ed_mu_ioniz = energy_dep[:,4]
  ed_mu_cut = energy_dep[:,5]
  ed_hadron_ioniz = energy_dep[:,6]
  ed_hadron_cut = energy_dep[:,7]
  ed_neutrino = energy_dep[:,8]
  ed_sum = energy_dep[:,9]


  ##############################################

  EnergyInNeutrinos = 1. # placeholder
  # + energy in all other particles

  # RawROOT addition
  EventName = "Event_" + str(EventID)

  AtmosphericModel = read_atmos(inp_input)
  Date = "2017-04-01" # from ATM file. TODO: unhardcode this
  print("[WARNING] date is hardcoded")
  t1 = time.strptime(Date.strip(),"%Y-%m-%d")
  UnixDate = int(time.mktime(t1))


  print("*****************************************")
  HadronicModel = "sibyll" #TODO:Unhardcode this
  print("[WARNING] hard-coded HadronicModel", HadronicModel) 
  LowEnergyModel = "urqmd" #TODO:Unhardcode this
  print("[WARNING] hard-coded LowEnergyModel", LowEnergyModel)
  print("*****************************************")

  # TODO: add function for reading logs
  # TODO: add CPU time
  CPUTime = 1.
  print("[WARNING] CPUTime is hardcoded")
  # TODO: find injection altitude in TPlotter.h/cpp
  InjectionAltitude = 100.
  print("[WARNING] InjectionAltitude is hardcoded")

  ArrayName = "GP13" # TODO: unhardcode this - do we even use this?
  # print("[WARNING] ArrayName is hardcoded")

  # SlantXmax
  # XmaxPosition
  # XmaxDistance
  # XmaxAltitude

  ############################################################################################################################
  # Part B.I.ii: Create and fill the RAW Shower Tree
  ############################################################################################################################
  OutputFileName = "Coreas_" + EventName +".root"

  # The tree with the Shower information common to ZHAireS and Coreas
  RawShower = RawTrees.RawShowerTree(OutputFileName)
  # The tree with Coreas-only info
  SimCoreasShower = RawTrees.RawCoreasTree(OutputFileName)
  # TODO: figure out what goes here -> its all leftover info 

  # ********** fill RawShower **********
  RawShower.run_number = RunID
  RawShower.sim_name = "Coreas"  # TODO:Unhardcode this, add version, etc
  RawShower.event_number = EventID
  RawShower.event_name = EventName
  RawShower.event_date = Date
  RawShower.unix_date = UnixDate

  RawShower.rnd_seed = RandomSeed[0] # TODO: figure out how to put the whole list here
  # right now I get this error "ValueError: setting an array element with a sequence." if I try to pass just "RandomSeed"

  RawShower.energy_in_neutrinos = EnergyInNeutrinos
  RawShower.energy_primary = [Energy]
  RawShower.azimuth = azimuth
  RawShower.zenith = zenith
  RawShower.primary_type = [str(Primary)]
  RawShower.primary_inj_alt_shc = [InjectionAltitude]
  RawShower.atmos_model = str(AtmosphericModel)

  RawShower.magnetic_field = np.array([FieldInclination,FieldDeclination,FieldIntensity])
  # RawShower.xmax_grams = SlantXmax
  # RawShower.xmax_pos_shc = XmaxPosition
  # RawShower.xmax_distance = XmaxDistance
  # RawShower.xmax_alt = XmaxAltitude
  RawShower.hadronic_model = HadronicModel
  RawShower.low_energy_model = LowEnergyModel
  RawShower.cpu_time = float(CPUTime)

  # * THINNING *
  RawShower.rel_thin = Thin[0]
  RawShower.maximum_weight = Thin[1]
  RawShower.hadronic_thinning = ThinH[0]
  RawShower.hadronic_thinning_weight = ThinH[1]
  RawShower.rmax = Thin[2]*100 #cm -> m

  # * CUTS *
  RawShower.lowe_cut_gamma = GammaEnergyCut
  RawShower.lowe_cut_e = ElectronEnergyCut
  RawShower.lowe_cut_mu = MuonEnergyCut
  RawShower.lowe_cut_meson = MesonEnergyCut # and hadrons
  RawShower.lowe_cut_nucleon = NucleonEnergyCut # same as meson and hadron cut

  RawShower.shower_core_pos = np.array(CorePosition)


  """
  In the next steps, fill the longitudinal profile, 
  i.e. the particle distribution ("pd") and energy deposit ("ed").
  
  These are matched with ZhaireS as good as possible. 
  Some fields will be missing here and some fields will be missing for ZhaireS.
  
  """

  RawShower.long_pd_gamma = pd_gammas
  RawShower.long_pd_eminus = pd_electrons
  RawShower.long_pd_eplus = pd_positrons
  RawShower.long_pd_muminus = pd_muN
  RawShower.long_pd_muplus = pd_muP
  RawShower.long_pd_allch = pd_charged
  RawShower.long_pd_nuclei = pd_nuclei
  RawShower.long_pd_hadr = pd_hadrons

  RawShower.long_ed_neutrino = ed_neutrino
  RawShower.long_ed_e_cut = ed_em_cut
  RawShower.long_ed_mu_cut = ed_mu_cut
  RawShower.long_ed_hadr_cut = ed_hadron_cut
  
  # gamma cut - I believe this was the same value as for another particle
  # for now: use hadron cut as placeholder
  # TODO ASAP: check this
  RawShower.long_ed_gamma_cut = ed_hadron_cut
  
  RawShower.long_ed_gamma_ioniz = ed_gamma
  RawShower.long_ed_e_ioniz = ed_em_ioniz
  RawShower.long_ed_mu_ioniz = ed_mu_ioniz
  RawShower.long_ed_hadr_ioniz = ed_hadron_ioniz
  
  # The next values are "leftover" from the comparison with ZhaireS.
  # They should go in TShowerSim along with the values above.
  RawShower.long_ed_depth = ed_depth
  RawShower.long_pd_depth = pd_depth
  
  RawShower.first_interaction = first_interaction

  RawShower.fill()
  RawShower.write()


  # *** fill MetaShower *** 


  #########################################################################################################################
  # Part B.II.i: get the information from Coreas output files (i.e. the traces and some extra info)
  #########################################################################################################################   
  
  #****** info from input files: ******

  # TODO ASAP: find these values
  RefractionIndexModel = "model"
  RefractionIndexParameters = [1,2,3] # ? 
  
  TimeWindowMin = TimeLowerBoundary # from reas
  TimeWindowMax = TimeUpperBoundary # from reas
  TimeBinSize   = TimeResolution    # from reas


  #****** load traces ******
  tracefiles = available_traces # from initial file checks

  #****** load positions ******
  # the list file contains all antenna positions for each antenna ID
  pathAntennaList = glob.glob(path + "*.list")[0]
  # store all antenna IDs in ant_IDs
  ant_IDs = antenna_positions_dict(pathAntennaList)["ID"]
  ############################################################################################################################
  # Part B.II.ii: Create and fill the RawEfield Tree
  ############################################################################################################################
 
  #****** fill shower info ******
  RawEfield = RawTrees.RawEfieldTree(OutputFileName)

  RawEfield.run_number = RunID
  RawEfield.event_number = EventID

  RawEfield.efield_sim = "Coreas" # TODO: unhardcode this and add versions

  RawEfield.refractivity_model = RefractionIndexModel                                       
  RawEfield.refractivity_model_parameters = RefractionIndexParameters                       
        
  RawEfield.t_pre = TimeWindowMin
  RawEfield.t_post = TimeWindowMax
  RawEfield.t_bin_size = TimeBinSize

  #****** fill traces ******
 
  RawEfield.du_count = len(tracefiles)

  # loop through polarizations and positions for each antenna
  print("******")
  print("filling traces")
  for index, antenna in enumerate(ant_IDs):
    tracefile = glob.glob(path + "SIM??????_coreas/raw_" + str(antenna) + ".dat")[0]

    # load the efield traces for this antenna
    # the files are setup like [timestamp, x polarization, y polarization, z polarization]
    efield = np.loadtxt(tracefile)
    
    timestamp = efield[:,0]
    trace_x = efield[:,1]
    trace_y = efield[:,2]
    trace_z = efield[:,3]
    
    # in Zhaires converter: AntennaN[ant_ID]
    RawEfield.du_id.append(int(antenna))
    RawEfield.t_0.append(timestamp[0].astype(float))

    # Traces
    RawEfield.trace_x.append(trace_x.astype(float))
    RawEfield.trace_y.append(trace_y.astype(float))
    RawEfield.trace_z.append(trace_z.astype(float))

    # Antenna positions in showers's referential in [m]
    ant_position = get_antenna_position(pathAntennaList, antenna)
    RawEfield.du_x.append(ant_position[0][index].astype(float))
    RawEfield.du_y.append(ant_position[1][index].astype(float))
    RawEfield.du_z.append(ant_position[2][index].astype(float))
    
  print("******")
  RawEfield.fill()
  RawEfield.write()
  #############################################################
  # fill SimCoreasShower with all leftover info               #
  #############################################################
  
  # store all leftover information here

  SimCoreasShower.AutomaticTimeBoundaries = AutomaticTimeBoundaries
  SimCoreasShower.ResolutionReductionScale = ResolutionReductionScale
  SimCoreasShower.GroundLevelRefractiveIndex = GroundLevelRefractiveIndex
  SimCoreasShower.GPSSecs = GPSSecs
  SimCoreasShower.GPSNanoSecs = GPSNanoSecs
  SimCoreasShower.DepthOfShowerMaximum = DepthOfShowerMaximum
  SimCoreasShower.DistanceOfShowerMaximum = DistanceOfShowerMaximum
  SimCoreasShower.GeomagneticAngle = GeomagneticAngle
  
  SimCoreasShower.nshow  = nshow # number of showers
  SimCoreasShower.ectmap = ectmap # does not affect output of sim - 100: every particle is printed in long file, 10E11 nothing is printed because cut is too high
  SimCoreasShower.maxprt = maxprt
  SimCoreasShower.radnkg = radnkg

  SimCoreasShower.parallel_ectcut = ECTCUT # = [ECTCUT, ECTMAX]
  SimCoreasShower.parallel_ectmax = ECTMAX

  SimCoreasShower.fill()
  SimCoreasShower.write()
  #############################################################

  print("### The event written was ", EventName, "###")
  print("### The name of the file is ", OutputFileName, "###")
  return EventName



if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    # make sure the last character is a slash
    if (path[-1]!="/"):
        path = path + "/"

    CoreasToRawRoot(path)
