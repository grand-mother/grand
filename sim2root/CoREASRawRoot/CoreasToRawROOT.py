#!/usr/bin/python
## Conversion of Coreas simulations to GRANDRaw ROOT files
## by Jelena Köhler, @jelenakhlr

import sys
import os
import glob
import time #to get the unix timestamp
from CorsikaInfoFuncs import * # this is in the same dir as this file
sys.path.append("../Common")
import raw_root_trees as RawTrees # this is in Common. since we're in CoREASRawRoot, this is in ../Common
from optparse import OptionParser

"""
run with
python3 CoreasToRawRoot <directory with Coreas Sim>

for more info, refer to the readme
"""

# add option parser to allow for reading either a single file or a full directory
parser = OptionParser()
parser.add_option("--directory", "--dir", "-d", type="str", dest="directory",
                  help="Specify the full path to the directory of the shower that you want to convert.")
parser.add_option("--file", "-f", type="str", dest="file",
                  help="Specify the full path to the SIMxxxxxx.reas file of the shower that you want to convert.")

(options, args) = parser.parse_args()

def CoreasToRawRoot(file, simID=None):
  print("-----------------------------------------")
  print("------ COREAS to RAWROOT converter ------")
  print("-----------------------------------------")
  ###############################
  # Part A: load Corsika files  #
  ###############################
  path = os.path.dirname(file)
  print("Checking", path, f"for SIM{simID}.reas and SIM/RUN{simID}.inp files (shower info).")
  
  # ********** load SIM.reas **********
  # load reas file
  reas_input = file

  # ********** load RUN.inp **********
  # find inp file
  # inp files can be named with SIM or RUN, so we will search for both
  if glob.glob(f"{path}/SIM{simID}.inp"):
    inp_input = f"{path}/SIM{simID}.inp"
  elif glob.glob(f"{path}/RUN{simID}.inp"):
    inp_input = f"{path}/RUN{simID}.inp"
  else:
     sys.exit("No input file found. Please check path and filename and try again.")

  # ********** load traces **********
  print("Checking subdirectories for *.dat files (traces).")
  available_traces = glob.glob("{path}/SIM{simID}_coreas/*.dat")
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
  log_file = glob.glob(f"{path}/DAT{simID}.log")

  if len(log_file) == 0:
    print("[WARNING] No log file found in this directory. Using dummy values for first interaction.")

    first_interaction = 100 # height of first interaction - in m
    print("Assuming first interaction at 100m.")
    hadr_interaction  = "Sibyll 2.3d"
    coreas_version    = "Coreas V1.4"
    print("Assuming hadronic interaction model Sibyll 2.3d and Coreas Version V1.4.")
  elif len(log_file) > 1:
    print("Found", log_file)
    print("[WARNING] More than one log file found in directory. Only log file", log_file[0], "will be used.")
    log_file = log_file[0]
    first_interaction = read_first_interaction(log_file) * 100 # height of first interaction - in m
    hadr_interaction  = read_HADRONIC_INTERACTION(log_file)
    coreas_version    = read_coreas_version(log_file)
  else:
    print("Found", log_file)
    log_file = log_file[0]
    print("Extracting info from log file", log_file, "for GRANDroot.")
    first_interaction = read_first_interaction(log_file) * 100 # height of first interaction - in m
    hadr_interaction  = read_HADRONIC_INTERACTION(log_file)
    coreas_version    = read_coreas_version(log_file)
  print("*****************************************")


  

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

  TimeResolution = read_params(reas_input, "TimeResolution") * 10**9 #convert to ns
  # TODO: add a check here to see if timeboundaries are auto or not
  AutomaticTimeBoundaries = read_params(reas_input, "AutomaticTimeBoundaries") * 10**9 #convert to ns
  TimeLowerBoundary = read_params(reas_input, "TimeLowerBoundary") * 10**9 # convert to ns
  TimeUpperBoundary = read_params(reas_input, "TimeUpperBoundary") * 10**9 # convert to ns
  ResolutionReductionScale = read_params(reas_input, "ResolutionReductionScale") * 100 # convert to m

  GroundLevelRefractiveIndex = read_params(reas_input, "GroundLevelRefractiveIndex") # refractive index at 0m asl

  RunID = int(read_params(reas_input, "RunNumber"))
  EventID = int(read_params(reas_input, "EventNumber"))
  GPSSecs = read_params(reas_input, "GPSSecs")
  GPSNanoSecs = read_params(reas_input, "GPSNanoSecs")
  FieldDeclination = read_params(reas_input, "RotationAngleForMagfieldDeclination") # in degrees

  if read_params(reas_input, "ShowerZenithAngle"):
    zenith = read_params(reas_input, "ShowerZenithAngle")
    azimuth = read_params(reas_input, "ShowerAzimuthAngle")

    Energy = read_params(reas_input, "PrimaryParticleEnergy") # in GeV
    Primary = read_params(reas_input, "PrimaryParticleType") # as defined in CORSIKA
    DepthOfShowerMaximum = read_params(reas_input, "DepthOfShowerMaximum") # slant depth in g/cm^2
    DistanceOfShowerMaximum = read_params(reas_input, "DistanceOfShowerMaximum") * 100 # geometrical distance of shower maximum from core in m
    FieldIntensity = read_params(reas_input, "MagneticFieldStrength") * 10**-1 # convert from Gauss to mT
    FieldInclination = read_params(reas_input, "MagneticFieldInclinationAngle") # in degrees, >0: in northern hemisphere, <0: in southern hemisphere
    GeomagneticAngle = read_params(reas_input, "GeomagneticAngle") # in degrees

  else:
    zenith = read_params(inp_input, "THETAP")
    azimuth = read_params(inp_input, "PHIP")

    Energy = read_params(inp_input, "ERANGE") # in GeV
    Primary = read_params(inp_input, "PRMPAR") # as defined in CORSIKA
    print("[WARNING] DepthOfShowerMaximum, DistanceOfShowerMaximum hardcoded")
    print("[WARNING] FieldIntensity, FieldInclination, GeomagneticAngle hardcoded for Dunhuang")
    DepthOfShowerMaximum = -1
    DistanceOfShowerMaximum = -1
    FieldIntensity = 0.5648236565
    FieldInclination = 61.60505071
    GeomagneticAngle = 93.82137564

  # from inp file
  nshow = read_params(inp_input, "NSHOW") # number of showers - should always be 1 for coreas, so maybe we dont need this parameter at all
  ectmap = str(read_params(inp_input, "ECTMAP"))
  maxprt = str(read_params(inp_input, "MAXPRT"))
  radnkg = str(read_params(inp_input, "RADNKG"))
  print("*****************************************")


  RandomSeed = read_params(inp_input, "SEED")

  ecuts = read_list_of_params(inp_input, "ECUTS")
  # 0: hadrons & nuclei, 1: muons, 2: e-, 3: photons
  GammaEnergyCut    = ecuts[3]
  ElectronEnergyCut = ecuts[2]
  MuonEnergyCut     = ecuts[1]
  HadronEnergyCut   = ecuts[0]
  NucleonEnergyCut  = ecuts[0]
  MesonEnergyCut    = HadronEnergyCut # mesons are hadronic, so this should be fine

  parallel = read_list_of_params(inp_input, "PARALLEL") # COREAS-only
  ECTCUT = parallel[0]
  ECTMAX = parallel[1]

  # PARALLEL = [ECTCUT, ECTMAX, MPIID, FECTOUT]
  # ECTCUT: limit for subshowers GeV
  # ECTMAX: maximum energy for complete shower GeV
  # MPIID: ID for mpi run (ignore for now)
  # T/F flag for extra output file (ignore for now)

  # In Zhaires converter: RelativeThinning, WeightFactor
  # I have:
  Thin  = read_list_of_params(inp_input, "THIN")
  # THIN = [limit, weight, Rmax]
  ThinH = read_list_of_params(inp_input, "THINH")
  # THINH = [limit, weight] for hadrons
  
  ##########################################
  # get all info from the long file
  pathLongFile = f"{path}/DAT{simID}.long"

  # the long file has an annoying setup, which I (very inelegantly) circumvent with this function:
  n_data, dE_data, hillas_parameter = read_long(pathLongFile)
  # there's an issue with the hillas_parameter in read_long, but also there seems to be a general issue with the hillas parameter in these files

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

  if EventID == 1:
     FileName = "Run_" + str(RunID)   
  else: 
     FileName = "Event_" + str(EventID)

  AtmosphericModel = read_atmos(inp_input)
  Date = read_date(inp_input)
  t1 = time.strptime(Date.strip(),"%Y-%m-%d")
  UnixDate = int(time.mktime(t1))


  print("*****************************************")
  HadronicModel = hadr_interaction
  LowEnergyModel = "urqmd" # might not be possible to get this info from mpi runs
  print("[WARNING] hard-coded LowEnergyModel", LowEnergyModel)
  print("*****************************************")

  # TODO: find injection altitude in TPlotter.h/cpp
  InjectionAltitude = 100.
  print("[WARNING] InjectionAltitude is hardcoded")

  ############################################################################################################################
  # Part B.I.ii: Create and fill the RAW Shower Tree
  ############################################################################################################################
  OutputFileName = "Coreas_" + FileName +".root"

  # The tree with the Shower information common to ZHAireS and Coreas
  RawShower = RawTrees.RawShowerTree(OutputFileName)
  # The tree with Coreas-only info
  SimCoreasShower = RawTrees.RawCoreasTree(OutputFileName)

  # ********** fill RawShower **********
  RawShower.run_number = RunID
  RawShower.sim_name = coreas_version
  RawShower.event_number = EventID
  RawShower.event_name = FileName
  RawShower.event_date = Date
  RawShower.unix_date = UnixDate

  RawShower.rnd_seed = RandomSeed

  RawShower.energy_in_neutrinos = EnergyInNeutrinos
  RawShower.energy_primary = [Energy]
  RawShower.azimuth = azimuth
  RawShower.zenith = zenith
  RawShower.primary_type = [str(Primary)]
  RawShower.primary_inj_alt_shc = [InjectionAltitude]
  RawShower.atmos_model = str(AtmosphericModel)

  RawShower.magnetic_field = np.array([FieldInclination,FieldDeclination,FieldIntensity])
  RawShower.hadronic_model = HadronicModel
  RawShower.low_energy_model = LowEnergyModel

  # * THINNING *
  RawShower.rel_thin = Thin[0]
  RawShower.maximum_weight = Thin[1]
  RawShower.hadronic_thinning = ThinH[0]
  RawShower.hadronic_thinning_weight = ThinH[1]
  RawShower.rmax = float(Thin[2]) * 100 #cm -> m

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

  RefractionIndexModel = "model"
  RefractionIndexParameters = [1,1,1] # ? 
  
  TimeBinSize   = TimeResolution    # from reas


  #****** load traces ******
  tracefiles = available_traces # from initial file checks

  #****** load positions ******
  # the list file contains all antenna positions for each antenna ID
  pathAntennaList = f"{path}/SIM{simID}.list"
  # store all antenna IDs in ant_IDs
  antenna_names = antenna_positions_dict(pathAntennaList)["name"]
  antenna_IDs   = antenna_positions_dict(pathAntennaList)["ID"] 

  ############################################################################################################################
  # Part B.II.ii: Create and fill the RawEfield Tree
  ############################################################################################################################
 
  #****** fill shower info ******
  RawEfield = RawTrees.RawEfieldTree(OutputFileName)

  RawEfield.run_number = RunID
  RawEfield.event_number = EventID

  RawEfield.refractivity_model = RefractionIndexModel                                       
  RawEfield.refractivity_model_parameters = RefractionIndexParameters                       
        
  RawEfield.t_bin_size = TimeBinSize

  #****** fill traces ******
 
  RawEfield.du_count = len(tracefiles)

  # loop through polarizations and positions for each antenna
  print("******")
  print("filling traces")
  for index, antenna in enumerate(antenna_names): 
    tracefile = f"{path}/SIM{simID}_coreas/raw_{str(antenna)}.dat"

    # load the efield traces for this antenna
    # the files are setup like [timestamp, x polarization, y polarization, z polarization]
    efield = np.loadtxt(tracefile)
    
    timestamp = efield[:,0] * 10**9 #convert to ns 
    trace_x = efield[:,1]
    trace_y = efield[:,2]
    trace_z = efield[:,3]
    
    # define time params:
    t_length = len(timestamp)
    t_0 = timestamp[0] + t_length/2
    t_pre = -t_length/2
    t_post = t_length/2

    # # timewindow min and max vary for each trace
    # TimeWindowMin = timestamp[0]
    # TimeWindowMax = timestamp[:-1]

    # RawEfield.TimeWindowMin.append(TimeWindowMin)
    # RawEfield.TimeWindowMax.append(TimeWindowMax)

    # in Zhaires converter: AntennaN[ant_ID]
    RawEfield.du_name.append(str(antenna))
    RawEfield.du_id.append(int(antenna_IDs[index]))

    # store time params:
    RawEfield.t_0.append(t_0.astype(float))
    RawEfield.t_pre = t_pre
    RawEfield.t_post = t_post

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

  SimCoreasShower.parallel_ectcut = ECTCUT 
  SimCoreasShower.parallel_ectmax = ECTMAX

  SimCoreasShower.fill()
  SimCoreasShower.write()
  #############################################################

  print("### The event written is", FileName, "###")
  print("### The name of the file is ", OutputFileName, "###")
  return FileName



if __name__ == "__main__":
  # * # * # * # * # * # * # * # * # * # *
  # convert multiple showers in one directory
  if options.directory:
    path = f"{options.directory}/"
    # find reas files in directory
    if glob.glob(path + "SIM??????.reas"):
        available_reas_files = glob.glob(path + "SIM??????.reas")
    else:
        print("No showers found. Please check your input and try again.")
        sys.exit()
    
    # get simIDs from the found reas files
    for reas_file in available_reas_files:
        shower_match = re.search(r'SIM(\d{6})\.reas', reas_file)
        if shower_match:
            simID = shower_match.group(1)
        else:
            print(f"No simID found for {reas_file}. Please check your input and try again.")
            sys.exit()
        CoreasToRawRoot(reas_file, simID)

  # * # * # * # * # * # * # * # * # * # *
  # convert a single shower
  elif options.file:
    file = options.file
    # find the simID of this file
    shower_match = re.search(r'SIM(\d{6})\.reas', file)
    if shower_match:
      simID = shower_match.group(1)
    else:
      print("Shower not found. Please check your input and try again.")
      sys.exit()
    # run the script
    CoreasToRawRoot(file, simID)

  # * # * # * # * # * # * # * # * # * # *
  # print help if options are not specified correctly
  else:
    parser.print_help()
