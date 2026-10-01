#!/usr/bin/env python
## Conversion of Coreas simulations to GRANDRaw ROOT files
## by Jelena Köhler, @jelenakhlr

import sys
import numpy as np
import os
import glob
import time #to get the unix timestamp
from CorsikaInfoFuncs import * # this is in the same dir as this file
# Run as a script from its own folder (as the README shows), the repository
# root is not on the path, and the sim2root.* imports below fail.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sim2root.Common.raw_root_trees as RawTrees # this is in Common. since we're in CoREASRawRoot, this is in ../Common
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
parser.add_option("--output", "-o", type="str", dest="output", default=".",
                  help="Folder to write Coreas_<simID>.rawroot into, or, for a single shower, the "
                       "file name. Default: the current folder.")
parser.add_option("--overwrite", action="store_true", dest="overwrite", default=False,
                  help="Replace an existing output file instead of refusing.")

(options, args) = parser.parse_args()


def check_antennas_and_traces(path_antenna_list, trace_dir):
  r"""Checks that the antenna list and the trace files describe the same, usable antennas.

  The converter used to take the antenna count from the trace files and
  everything else from the list, so a truncated list, a missing trace or an
  extra one produced a file whose ``du_count`` disagreed with its traces, and
  NaN or truncated traces were written as they were (issue #243).

  Parameters
  ----------
  path_antenna_list : str
      The ``SIM<id>.list`` file.
  trace_dir : str
      The ``SIM<id>_coreas`` folder holding one ``raw_<name>.dat`` per antenna.

  Raises
  ------
  ValueError
      Naming every problem found: duplicate names or IDs, antennas without a
      trace, traces without an antenna, non-finite positions, and traces that
      are not finite, not 4 columns, or of different lengths.
  """
  problems = []
  info = antenna_positions_dict(path_antenna_list)
  names = [str(n) for n in info["name"]]
  if len(set(names)) != len(names):
    problems.append("duplicate antenna names: %s" % sorted({n for n in names if names.count(n) > 1})[:10])
  ids = list(info["ID"])
  if len(set(ids)) != len(ids):
    problems.append("duplicate antenna IDs: %s" % sorted({i for i in ids if ids.count(i) > 1})[:10])
  if not all(np.all(np.isfinite(info[axis])) for axis in ("x", "y", "z")):
    problems.append("antenna positions that are not finite numbers")
  traces = {os.path.basename(f)[len("raw_"):-len(".dat")]
            for f in glob.glob(os.path.join(trace_dir, "raw_*.dat"))}
  missing = sorted(set(names) - traces)
  extra = sorted(traces - set(names))
  if missing:
    problems.append("%d listed antennas have no trace file, e.g. %s" % (len(missing), missing[:10]))
  if extra:
    problems.append("%d trace files are not in the antenna list, e.g. %s" % (len(extra), extra[:10]))
  lengths = set()
  for name in sorted(set(names) & traces):
    trace = np.loadtxt(os.path.join(trace_dir, "raw_%s.dat" % name), ndmin=2)
    if trace.ndim != 2 or trace.shape[1] != 4:
      problems.append("raw_%s.dat has shape %s, expected (samples, 4)" % (name, trace.shape))
      continue
    if not np.all(np.isfinite(trace)):
      problems.append("raw_%s.dat holds NaN or infinite values" % name)
    lengths.add(trace.shape[0])
  if len(lengths) > 1:
    problems.append("traces have different lengths: %s samples" % sorted(lengths))
  if problems:
    raise ValueError("GRANDlib: CoreasToRawROOT: %s and %s do not match:\n  - %s"
                     % (path_antenna_list, trace_dir, "\n  - ".join(problems)))

def output_file(output, RunID, overwrite=False):
  r"""Returns the RawROOT file to write, refusing an existing one (#181).

  The trees open their file for appending, so a second conversion of the
  same shower -- or the committed sample in this folder -- failed with
  NotUniqueEvent.  An existing file is now refused, or replaced with
  ``overwrite``.
  """
  if output.endswith(".rawroot"):
    name = output
  else:
    name = os.path.join(output, "Coreas_" + RunID + ".rawroot")
  if os.path.exists(name):
    if not overwrite:
      sys.exit("GRANDlib: CoreasToRawROOT: %s already exists; remove it or use --overwrite" % name)
    os.remove(name)
  parent = os.path.dirname(name)
  if parent:
    os.makedirs(parent, exist_ok=True)
  return name


def CoreasToRawRoot(file, simID=None, output=".", overwrite=False):
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
  print(f"Found {file}")

  # ********** load RUN.inp **********
  # find inp file
  # inp files can be named with SIM or RUN, so we will search for both
  if glob.glob(f"{path}/SIM{simID}.inp"):
    inp_input = f"{path}/SIM{simID}.inp"
    print(f"Found {inp_input}")
  elif glob.glob(f"{path}/RUN{simID}.inp"):
    inp_input = f"{path}/RUN{simID}.inp"
    print(f"Found {inp_input}")
  else:
     sys.exit("No input file found. Please check path and filename and try again.")

  # ********** load traces **********
  print("Checking subdirectories for *.dat files (traces).")
  available_traces = glob.glob(f"{path}/SIM{simID}_coreas/*.dat")
  if len(available_traces) == 0:
    sys.exit("No traces found. Please check path and try again.")
  else:
    print("Found", len(available_traces), "*.dat files (traces).")
  # The antenna list and the trace files must describe the same antennas, and
  # every trace must be usable, before anything is written (issue #243).
  check_antennas_and_traces(f"{path}/SIM{simID}.list", f"{path}/SIM{simID}_coreas")
     
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

    first_interaction = 1 # height of first interaction - in m
    print("[WARNING] Assuming first interaction at 1m.")
    hadr_interaction  = "Sibyll 2.3d"
    coreas_version    = "1.4"
    print("Assuming hadronic interaction model Sibyll 2.3d and Coreas Version V1.4.")
  elif len(log_file) > 1:
    print("Found", log_file)
    print("[WARNING] More than one log file found in directory. Only log file", log_file[0], "will be used.")
    log_file = log_file[0]
    first_interaction = read_first_interaction(log_file) / 100 # height of first interaction - in m
    hadr_interaction  = read_HADRONIC_INTERACTION(log_file)
    coreas_version    = read_coreas_version(log_file)
  else:
    print("Found", log_file)
    log_file = log_file[0]
    print("Extracting info from log file", log_file, "for GRANDroot.")
    first_interaction = read_first_interaction(log_file) / 100 # height of first interaction - in m
    hadr_interaction  = read_HADRONIC_INTERACTION(log_file)
    coreas_version    = read_coreas_version(log_file)

  corsika_version = read_corsika_version(inp_input)

  
  print("*****************************************")


  

  ###############################
  # Part B: Generate ROOT Trees #
  ###############################

  #########################################################################################################################
  # Part B.I.i: get the information from Coreas input files
  #########################################################################################################################   
  # from reas file
  CoreCoordinateNorth = read_params(reas_input, "CoreCoordinateNorth") / 100 # convert to m
  CoreCoordinateWest = read_params(reas_input, "CoreCoordinateWest") / 100 # convert to m
  CoreCoordinateVertical = read_params(reas_input, "CoreCoordinateVertical") / 100 # convert to m
  CorePosition = [CoreCoordinateNorth, CoreCoordinateWest, CoreCoordinateVertical]

  TimeResolution = read_params(reas_input, "TimeResolution") * 10**9 #convert to ns
  if not TimeResolution > 0:
    raise ValueError("GRANDlib: CoreasToRawROOT: TimeResolution in %s must be positive, got %s ns"
                     % (reas_input, TimeResolution))
  # TODO: add a check here to see if timeboundaries are auto or not
  AutomaticTimeBoundaries = read_params(reas_input, "AutomaticTimeBoundaries") * 10**9 #convert to ns
  TimeLowerBoundary = read_params(reas_input, "TimeLowerBoundary") * 10**9 # convert to ns
  TimeUpperBoundary = read_params(reas_input, "TimeUpperBoundary") * 10**9 # convert to ns
  ResolutionReductionScale = read_params(reas_input, "ResolutionReductionScale") / 100 # convert to m

  GroundLevelRefractiveIndex = read_params(reas_input, "GroundLevelRefractiveIndex") # refractive index at 0m asl

  RunID = simID
  EventID = int(read_params(reas_input, "EventNumber"))
  print("[WARNING] dummy values for GPSSecs and GPSNanoSecs")
  GPSSecs = 1996#read_params(reas_input, "GPSSecs")
  GPSNanoSecs = 19961026#read_params(reas_input, "GPSNanoSecs")
  FieldDeclination = read_params(reas_input, "RotationAngleForMagfieldDeclination") # in degrees

  # `read_params` returns None for a keyword the file does not carry, so the
  # question here is "is the card present", not "is it non-zero".  Tested for
  # truth, a genuine ShowerZenithAngle of 0.0 -- a vertical shower -- is
  # falsy and silently takes the branch below, which discards the file's own
  # parameters in favour of hard-coded Dunhuang values.
  if read_params(reas_input, "ShowerZenithAngle") is not None:
    zenith = read_params(reas_input, "ShowerZenithAngle")
    # CoREAS gives the direction of travel; GRAND gives where the shower comes from
    azimuth = (read_params(reas_input, "ShowerAzimuthAngle") + 180) % 360

    Energy = read_params(reas_input, "PrimaryParticleEnergy") * 1e-9 # in GeV
    Primary = read_params(reas_input, "PrimaryParticleType") # as defined in CORSIKA
    DepthOfShowerMaximum = read_params(reas_input, "DepthOfShowerMaximum") # slant depth in g/cm^2
    DistanceOfShowerMaximum = read_params(reas_input, "DistanceOfShowerMaximum") / 100 # geometrical distance of shower maximum from core in m
    FieldIntensity = read_params(reas_input, "MagneticFieldStrength") * 10 ** (-1) # convert from Gauss to mT
    FieldInclination = read_params(reas_input, "MagneticFieldInclinationAngle") # in degrees, >0: in northern hemisphere, <0: in southern hemisphere
    GeomagneticAngle = read_params(reas_input, "GeomagneticAngle") # in degrees

    # CoREAS writes -1 for "not known" (issue #228): taken as a distance of
    # -1 cm it put Xmax 1 cm from the core, which made every voltage ~1e-13 uV.
    # Unknown is NaN, as on the .inp path below.
    if DistanceOfShowerMaximum is None or not DistanceOfShowerMaximum > 0:
      print("[WARNING] DistanceOfShowerMaximum is not given (%s); Xmax position written as NaN"
            % DistanceOfShowerMaximum)
      DistanceOfShowerMaximum = np.nan
    if DepthOfShowerMaximum is None or not DepthOfShowerMaximum > 0:
      DepthOfShowerMaximum = np.nan

    # calculate Xmax cartesian position
    # set spherical system vector in m and radians
    Xmax_sph = np.array([DistanceOfShowerMaximum, np.deg2rad(zenith), np.deg2rad(azimuth)])
    # simply transform from spherical to cartesian to recover the cartesian position of Xmax
    Xmax_NWU = np.array([Xmax_sph[0] * np.sin(Xmax_sph[1]) * np.cos(Xmax_sph[2]), \
                         Xmax_sph[0] * np.sin(Xmax_sph[1]) * np.sin(Xmax_sph[2]), \
                         Xmax_sph[0] * np.cos(Xmax_sph[1])])

  else:
    #theta_GRAND = theta_Corsika
    zenith = read_params(inp_input, "THETAP")
    # CORSIKA's PHIP is the azimuth of the primary's momentum (direction of
    # travel), in a frame with x North and y West like GRAND's.  The "comes
    # from" azimuth is therefore PHIP - 180, as in the .reas branch above.
    # It used to be 180 - PHIP, which mirrored every shower about North
    # (issue #209).
    azimuth = (read_params(inp_input, "PHIP") - 180) % 360

    Energy = read_params(inp_input, "ERANGE") # in GeV
    Primary = read_params(inp_input, "PRMPAR") # as defined in CORSIKA
    print("[WARNING] DepthOfShowerMaximum, DistanceOfShowerMaximum hardcoded")
    print("[WARNING] FieldIntensity, FieldInclination, GeomagneticAngle hardcoded for Dunhuang")
    DepthOfShowerMaximum = -1
    DistanceOfShowerMaximum = -1
    FieldIntensity = 0.5648236565
    FieldInclination = 61.60505071
    GeomagneticAngle = 93.82137564

    # Xmax's cartesian position is derived from DistanceOfShowerMaximum, which
    # this branch does not have (-1 above, meaning "unknown").  The branch
    # above defines Xmax_NWU and the write below is unconditional, so leaving
    # the name undefined here aborts the conversion partway through with
    # `UnboundLocalError: Xmax_NWU` -- which is what the repository's own
    # CoREAS fixture does (issue #159).
    #
    # NaN rather than zeros: a zero vector reads downstream as a real Xmax
    # sitting at the array origin, and nothing between here and a plot range
    # checks it.  NaN propagates visibly instead of being averaged in.
    print("[WARNING] Xmax position unavailable on this path "
          "(no DistanceOfShowerMaximum); writing NaN")
    Xmax_NWU = np.full(3, np.nan)

  # from inp file
  nshow = read_params(inp_input, "NSHOW") # number of showers - should always be 1 for coreas, so maybe we dont need this parameter at all
  ectmap = str(read_params(inp_input, "ECTMAP"))
  maxprt = str(read_params(inp_input, "MAXPRT"))
  radnkg = str(read_params(inp_input, "RADNKG"))
  print("*****************************************")


  RandomSeed = read_params(inp_input, "SEED")

  ecuts = read_required_list_of_params(inp_input, "ECUTS")
  # 0: hadrons & nuclei, 1: muons, 2: e-, 3: photons
  GammaEnergyCut    = ecuts[3]
  ElectronEnergyCut = ecuts[2]
  MuonEnergyCut     = ecuts[1]
  HadronEnergyCut   = ecuts[0]
  NucleonEnergyCut  = ecuts[0]
  MesonEnergyCut    = HadronEnergyCut # mesons are hadronic, so this should be fine

  parallel = read_list_of_params(inp_input, "PARALLEL") # COREAS-only
  if parallel is None:
    # A non-parallel CoREAS run writes no PARALLEL card. That is not an error;
    # the two fields simply have no value, and -1 is this converter's "not
    # available". Resolves issue #147.
    print("[WARNING] No PARALLEL found in inp file. Setting ECTCUT and ECTMAX to -1.")
    ECTCUT = -1
    ECTMAX = -1
  else:
    ECTCUT = parallel[0]
    ECTMAX = parallel[1]
  
  # PARALLEL = [ECTCUT, ECTMAX, MPIID, FECTOUT]
  # ECTCUT: limit for subshowers GeV
  # ECTMAX: maximum energy for complete shower GeV
  # MPIID: ID for mpi run (ignore for now)
  # T/F flag for extra output file (ignore for now)

  # In Zhaires converter: RelativeThinning, WeightFactor
  # I have:
  Thin  = read_required_list_of_params(inp_input, "THIN")
  # THIN = [limit, weight, Rmax]
  ThinH = read_required_list_of_params(inp_input, "THINH")
  # THINH = [limit, weight] for hadrons
  
  ##########################################
  # get all info from the long file
  pathLongFile = f"{path}/DAT{simID}.long"

  # the long file has an annoying setup, which I (very inelegantly) circumvent with this function:
  n_data, dE_data, hillas_parameters = read_long(pathLongFile)
  
  Xmax = hillas_parameters[0] # Coreas & GRAND: g/cm^2
  Chi_hillas = hillas_parameters[1]

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


  Egamma = np.sum(ed_gamma)
  Eem_ion = np.sum(ed_em_ioniz)
  Eem_cut = np.sum(ed_em_cut)

  # calculate electromagnetic shower energy and leave them in GeV
  Eem = (Egamma + Eem_ion + Eem_cut) # in GeV
  print("Electromagnetic shower energy:", Eem)


  ##############################################

  EnergyInNeutrinos = 1. # placeholder
  # + energy in all other particles

  AtmosphericModel = read_atmos(inp_input)
  Date = read_date(inp_input)
  t1 = time.strptime(Date.strip(),"%Y-%m-%d")
  UnixDate = int(time.mktime(t1))


  print("*****************************************")
  HadronicModel = hadr_interaction
  LowEnergyModel = "urqmd" # might not be possibleV to get this info from mpi runs
  print("[WARNING] hard-coded LowEnergyModel", LowEnergyModel)
  print("*****************************************")

  # TODO: find injection altitude in TPlotter.h/cpp
  InjectionAltitude = 1.
  print("[WARNING] InjectionAltitude is hardcoded")

  site = read_site(inp_input)
  latitude, longitude, altitude = read_lat_long_alt(site)

  # set altitude to simulations obslevel
  altitude = CorePosition[2]

  ############################################################################################################################
  # Part B.I.ii: Create and fill the RAW Shower Tree
  ############################################################################################################################
  OutputFileName = output_file(output, RunID, overwrite)

  # The tree with the Shower information common to ZHAireS and Coreas
  RawShower = RawTrees.RawShowerTree(OutputFileName)
  # The tree with Coreas-only info
  SimCoreasShower = RawTrees.RawCoreasTree(OutputFileName)

  # ********** fill RawShower **********
  RawShower.run_number = EventID
  RawShower.sim_name = str("Corsika")
  RawShower.sim_version = str(corsika_version)
  RawShower.event_number = int(RunID)  # the ID is a string such as "004100"
  RawShower.event_name = RunID
  RawShower.event_date = Date
  RawShower.unix_date = UnixDate

  RawShower.rnd_seed = RandomSeed

  RawShower.energy_in_neutrinos = EnergyInNeutrinos
  RawShower.energy_em = [Eem]
  RawShower.energy_primary = [Energy]
  RawShower.azimuth = azimuth
  RawShower.zenith = zenith
  RawShower.primary_type = [str(Primary)]
  RawShower.primary_inj_alt_shc = [InjectionAltitude]
  RawShower.atmos_model = str(AtmosphericModel)
  RawShower.site = site
  RawShower.magnetic_field = np.array([FieldInclination,FieldDeclination,FieldIntensity])
  RawShower.hadronic_model = HadronicModel
  RawShower.low_energy_model = LowEnergyModel

  #* site specs *
  RawShower.site = site
  RawShower.site_lat = latitude
  RawShower.site_lon = longitude
  RawShower.site_alt = altitude

  # * THINNING *
  RawShower.rel_thin = Thin[0]
  RawShower.maximum_weight = Thin[1]
  RawShower.hadronic_thinning = ThinH[0]
  RawShower.hadronic_thinning_weight = ThinH[1]
  RawShower.rmax = float(Thin[2]) * 10**-2 #cm -> m

  # * CUTS *
  RawShower.lowe_cut_gamma = GammaEnergyCut
  RawShower.lowe_cut_e = ElectronEnergyCut
  RawShower.lowe_cut_mu = MuonEnergyCut
  RawShower.lowe_cut_meson = MesonEnergyCut # and hadrons
  RawShower.lowe_cut_nucleon = NucleonEnergyCut # same as meson and hadron cut

  


  """
  In the next steps, fill the longitudinal profile, 
  i.e. the particle distribution ("pd") and energy deposit ("ed").
  
  These are matched with ZhaireS as good as possible. 
  Some fields will be missing here and some fields will be missing for ZhaireS.
  
  """
  RawShower.xmax_grams = Xmax
  RawShower.xmax_distance = DistanceOfShowerMaximum
  RawShower.xmax_pos_shc = Xmax_NWU

  # RawShowerTree calls it long_pd_gammas; the gamma profile was stored nowhere (#202)
  RawShower.long_pd_gammas = pd_gammas
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

  pathAntennaList = f"{path}/SIM{simID}.list"
  core_shift_x, core_shift_y = calculate_array_shift(pathAntennaList)
  
  #the array coordinate system has its origin at ground level
  print("[WARNING] antenna altitude is hard coded to 0")
  RawShower.shower_core_pos = np.array(CorePosition) + np.array([core_shift_x, core_shift_y,-CorePosition[2]])
  print("SHOWER CORE POS:", RawShower.shower_core_pos)
  RawShower.fill()
  RawShower.write()


  # *** fill MetaShower *** 
  ###########################################################################################################################
  # Part B.II.iii: Create and fill the RawMetaEfield Tree
  ############################################################################################################################
  #There are facilities in Common to build an EventParameters file with all this info and then read the info from the file
  #It will come handy in the future if you test several core positions before running the sim, for storing the tested core positions
  #and giving a weight to the event.
  print("***RawMeta***")
    
  RawMeta = RawTrees.RawMetaTree(OutputFileName)
  RawMeta.run_number = int(RunID)
  RawMeta.event_number = EventID

  print("[WARNING] array_name is hardcoded")
  RawMeta.array_name = "GP300"
  RawMeta.shower_core_pos=RawShower.shower_core_pos  #MT this is twice in the file, i have the idea it should only be here (for me the core position is "meta" to the shower sim, but it looks like is an input for coreas, so i keep both)
  RawMeta.unix_second=GPSSecs
  RawMeta.unix_nanosecond=GPSNanoSecs
  print("[WARNING] event_weight is hardcoded")  
  RawMeta.event_weight=1 
  print("******")
  RawMeta.fill()
  RawMeta.write()  


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
  # stores the actual antenna names from the coreas -list file
  antenna_names = antenna_positions_dict(pathAntennaList)["name"]
  # if antenna name is in the form of "ant???" or "gp_???", stores the digits after "ant" or "gp_", otherwise generic counter from 1
  antenna_IDs = antenna_positions_dict(pathAntennaList)["ID"] 

  ############################################################################################################################
  # Part B.II.ii: Create and fill the RawEfield Tree
  ############################################################################################################################
 
  #****** fill shower info ******
  RawEfield = RawTrees.RawEfieldTree(OutputFileName)

  RawEfield.run_number = EventID
  RawEfield.event_number = int(RunID)
  RawEfield.sim_name = str("CoREAS")
  RawEfield.sim_version = str(coreas_version)

  RawEfield.refractivity_model = RefractionIndexModel                                       
  RawEfield.refractivity_model_parameters = RefractionIndexParameters                       
        
  RawEfield.t_bin_size = TimeBinSize

  #****** fill traces ******
 
  # One entry per listed antenna; check_antennas_and_traces made sure each has its trace
  RawEfield.du_count = len(antenna_names)

  # loop through polarizations and positions for each antenna
  print("******")
  print("filling traces")
  for index, antenna in enumerate(antenna_names): 
    tracefile = f"{path}/SIM{simID}_coreas/raw_{str(antenna)}.dat"

    # load the efield traces for this antenna
    # the files are setup like [timestamp, x polarization, y polarization, z polarization]
    efield = np.loadtxt(tracefile)
    
    timestamp = efield[:,0] * 10**9 # convert to ns 
    # coreas uses cgs units, so voltage is in statvolt
    # efield in statvolt / cm
    # 1 statV * 299.792458 V/statV = 1 V
    # 1 statV * 299.792458 * 1e6 V/statV = 1 microV
    trace_x = efield[:,1]* 299.792458 * 1e6 * 100 # convert to micro Volts / m
    trace_y = efield[:,2]* 299.792458 * 1e6 * 100 # convert to micro Volts / m
    trace_z = efield[:,3]* 299.792458 * 1e6 * 100 # convert to micro Volts / m



    # define time params:
    t_length = len(timestamp)
    t_0 = timestamp[0] + t_length/2 * TimeBinSize
    t_pre  = 800#ns
    t_pre = t_length/2 * TimeBinSize
    t_post = t_length/2 * TimeBinSize
    
    # add to ROOT tree
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
    ant_position_x, ant_position_y, ant_position_z = get_antenna_position(pathAntennaList, antenna)
    # shift antenna z coordinate to array level
    ant_position_z = ant_position_z - altitude
    RawEfield.du_x.append(ant_position_x.astype(float))
    RawEfield.du_y.append(ant_position_y.astype(float))
    RawEfield.du_z.append(ant_position_z.astype(float))
    
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

  print("### The event written is", RunID, "###")
  print("### The name of the file is ", OutputFileName, "###")
  return RunID


if __name__ == "__main__":
  # * # * # * # * # * # * # * # * # * # *
  # convert multiple showers in one directory
  if options.directory:
    path = f"{options.directory}/"
    # find reas files in directory
    available_reas_files = glob.glob(path + "SIM??????.reas")
    if not available_reas_files:
        sys.exit("Error: No showers found in the specified directory. Please check your input and try again.")
    
    if options.output.endswith(".rawroot") and len(available_reas_files) > 1:
        sys.exit("GRANDlib: CoreasToRawROOT: %d showers in %s; -o must name a folder, not a file"
                 % (len(available_reas_files), options.directory))
    # get simIDs from the found reas files
    for reas_file in sorted(available_reas_files):
        shower_match = re.search(r'SIM(\d{6})\.reas', reas_file)
        if shower_match:
            simID = shower_match.group(1)
            print(f"Run number: {simID}")
        else:
            sys.exit(f"Error: No simID found for {reas_file}. Please check your input and try again.")
        CoreasToRawRoot(reas_file, simID, options.output, options.overwrite)

  # * # * # * # * # * # * # * # * # * # *
  # convert a single shower
  elif options.file:
    file = options.file
    # find the simID of this file
    shower_match = re.search(r'SIM(\d{6})\.reas', file)
    if shower_match:
      simID = shower_match.group(1)
    else:
      sys.exit("Error: Shower not found in the specified file. Please check your input and try again.")
    # run the script
    CoreasToRawRoot(file, simID, options.output, options.overwrite)

  # * # * # * # * # * # * # * # * # * # *
  # print help if options are not specified correctly
  else:
    print("Error: No valid options specified. Please provide either a directory or a file to convert.")
    parser.print_help()
    sys.exit(2)
    sys.exit()
