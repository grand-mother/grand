#!/usr/bin/python
## Conversion of GRANDRaw ROOT files coming from simulators to GRANDROOT files
## by Lech Wiktor Piotrowski

import os
import argparse
from types import SimpleNamespace
import time
from pathlib import Path
from grand.dataio.root_trees import * # this is home/grand/grand (at least in docker) or ../../grand
import raw_root_trees as RawTrees # this is here in Common


#ToDo:latitude,longitude and altitude are available in ZHAireS .sry file, and could be added to the RawRoot file. Site too.
# Command line argument parsing
clparser = argparse.ArgumentParser(description="Convert simulation data in rawroot format into GRANDROOT format")
clparser.add_argument("filename", nargs='+', help="ROOT file containing GRANDRaw data TTrees")
clparser.add_argument("-o", "--output_parent_directory", help="Output parent directory", default="")
clparser.add_argument("-fo", "--forced_output_directory", help="Force this option as the output directory", default=None)
clparser.add_argument("-s", "--site_name", help="The name of the site", default="nosite")
clparser.add_argument("-d", "--sim_date", help="The date of the simulation", default=None)
clparser.add_argument("-t", "--sim_time", help="The time of the simulation", default=None)
# clparser.add_argument("-d", "--sim_date", help="The date of the simulation", default="19000101")
# clparser.add_argument("-t", "--sim_time", help="The time of the simulation", default="000000")
clparser.add_argument("-e", "--extra", help="Extra information to store in the directory name", default="")
clparser.add_argument("-av", "--analysis_level", help="Analysis level of the data", default=0, type=int)
# clparser.add_argument("-se", "--serial", help="Serial number of the simulation", default=0)
clparser.add_argument("-la", "--latitude", help="Latitude of the site", default=40.984558)
clparser.add_argument("-lo", "--longitude", help="Longitude of the site", default=93.952247)
clparser.add_argument("-al", "--altitude", help="Altitude of the site", default=1200)
clparser.add_argument("-ru", "--run", help="Run number", default=0, const=None)
clparser.add_argument("-se", "--start_event", help="Starting event number", default=0, const=None)
clargs = clparser.parse_args()

print("#################################################")

def convert_date(date_str):
    # Convert input string to a struct_time object
    date_struct = time.strptime(date_str, "%Y-%m-%d")
    # Format the struct_time object as a string in YYYYMMDD format
    formatted_date = time.strftime("%Y%m%d", date_struct)
    return formatted_date


def main():
    # Initialise the run number if specified
    ext_run_number = None
    if clargs.run is not None:
        ext_run_number = int(clargs.run)

    # Initialise the starting event number
    ext_event_number = None
    if clargs.start_event is not None:
        ext_event_number = int(clargs.start_event)

    start_event_number = 0

    # Namespace for holding output trees
    gt = SimpleNamespace()

    # Loop through the files specified on command line
    for file_num, filename in enumerate(clargs.filename):

        # Output filename for GRAND Trees
        # if clargs.output_filename is None:
        out_filename = os.path.join(os.path.split(filename)[0], "gr_"+os.path.split(filename)[1])
        # else:
        #     out_filename = clargs.output_filename

        # Read the raw trees from the file
        trawshower = RawTrees.RawShowerTree(filename)
        trawefield = RawTrees.RawEfieldTree(filename)
        trawmeta = RawTrees.RawMetaTree(filename)

        # Loop through entries - assuming same number of entries in each tree
        # ToDo: this should be a tree iterator through one tree and getting the other through friends. Need to make friends working...
        nentries = trawshower.get_entries()
        for i in range(nentries):
            trawshower.get_entry(i)
            trawefield.get_entry(i)
            trawmeta.get_entry(i)

            # If the first entry or (run number enforced and first file), fill the run trees
            if (i==0 and ext_run_number is None) or (ext_run_number is not None and file_num==0):

                # Overwrite the run number if specified on command line
                run_number = ext_run_number if ext_run_number is not None else gt.trun.run_number

                # Init output trees in the proper directory
                out_dir_name = init_trees(clargs, trawshower.unix_date, run_number, gt)

                # Convert the RawShower entries
                rawshower2grandrootrun(trawshower, gt)
                # Convert the RawEfield entries
                rawefield2grandrootrun(trawefield, gt)

                #ToDo:latitude,longitude and altitude are available in ZHAireS .sry file, and could be added to the RawRoot file

                # Set the origin geoid
                gt.trun.origin_geoid = get_origin_geoid(clargs)

                gt.trun.run_number = run_number
                gt.trunshowersim.run_number = run_number
                gt.trunefieldsim.run_number = run_number

                # Fill the run trees and write
                # gt.trun.fill()
                gt.trunshowersim.fill()
                gt.trunefieldsim.fill()
                # gt.trun.write()
                gt.trunshowersim.write()
                gt.trunefieldsim.write()

            # Convert the RawShowerTree entries
            rawshower2grandroot(trawshower, gt)
            # Convert the RawMetaTree entries - (this goes before the efield becouse the efield needs the info on the second and nanosecond)
            rawmeta2grandroot(trawmeta, gt)
            # Convert the RawEfieldTree entries
            rawefield2grandroot(trawefield, gt)

            # Overwrite the run number if specified on command line
            if ext_run_number is not None:
                gt.trun.run_number = ext_run_number
                gt.trunshowersim.run_number = ext_run_number
                gt.trunefieldsim.run_number = ext_run_number
                gt.tshower.run_number = ext_run_number
                gt.tshowersim.run_number = ext_run_number
                gt.tefield.run_number = ext_run_number

            # Overwrite the event number if specified on command line
            if ext_event_number is not None:
                gt.tshower.event_number = ext_event_number
                gt.tshowersim.event_number = ext_event_number
                gt.tefield.event_number = ext_event_number

            # For the first file/iteration, store the event number
            if file_num==0 and i==0:
                start_event_number = gt.tshower.event_number

            # Fill the event trees
            gt.tshower.fill()
            gt.tshowersim.fill()
            gt.tefield.fill()

        # For the first file, get all the file's events du ids and pos
        if file_num==0:
            du_ids, du_xyzs = get_tree_du_id_and_xyz(trawefield)
        # For other files, append du ids and pos to the ones already retrieved
        else:
            tdu_ids, tdu_xyzs = get_tree_du_id_and_xyz(trawefield)
            du_ids = np.append(du_ids, tdu_ids)
            du_xyzs = np.vstack([du_xyzs, tdu_xyzs])

        # Write the event trees
        gt.tshower.write()
        gt.tshowersim.write()
        gt.tefield.write()
        # gt.tshower.first_interaction = trawshower.first_interaction

        # Increment the event number if starting one specified on command line
        if ext_event_number is not None:
            ext_event_number += 1

    # Fill the trun with antenna positions and ids from ALL the events
    # ToDo: this should be done with TChain in one loop over all the files... maybe (which would be faster?)

    # Get indices of the unique du_ids
    unique_dus_idx = np.unique(du_ids, return_index=True)[1]
    # Leave only the unique du_ids
    du_ids = du_ids[unique_dus_idx]
    # Sort the DUs
    sorted_idx = np.argsort(du_ids)
    du_ids = du_ids[sorted_idx]
    # Stack x/y/z together and leave only the ones for unique du_ids, sort
    du_xyzs = du_xyzs[unique_dus_idx][sorted_idx]

    # Assign the du ids and positions to the trun tree
    gt.trun.du_id = du_ids
    gt.trun.du_xyz = du_xyzs

    # ToDo: shouldn't this and above be created for every DU in sims?
    gt.trun.t_bin_size = [trawefield.t_bin_size]*len(du_ids) #Matias Question: Why is this being mutiplied here?

    # Fill and write the TRun
    gt.trun.fill()
    gt.trun.write()


    # Rename the created files to appropriate names
    end_event_number = gt.tshower.event_number
    print("Renaming files to proper file names")
    rename_files(clargs, out_dir_name, start_event_number, end_event_number)

# Initialise output trees and their directory
def init_trees(clargs, unix_date, run_number, gt):

    # Use date/time from command line argument if specified, otherwise the unix time
    date, time = datetime.datetime.utcfromtimestamp(unix_date).strftime('%Y%m%d_%H%M%S').split("_")
    if clargs.sim_date is not None:
        date = clargs.sim_date
    if clargs.sim_time is not None:
        time = clargs.sim_time

    # Create the appropriate output directory
    if clargs.forced_output_directory is None:
        out_dir_name = form_directory_name(clargs, date, time, run_number)
        print("Storing files in directory ", out_dir_name)
        out_dir_name.mkdir()
    # If another directory was forced as the output directory, create it
    else:
        out_dir_name = Path(clargs.output_parent_directory, clargs.forced_output_directory)
        out_dir_name.mkdir(exist_ok=True)

    # Create appropriate GRANDROOT trees in temporary file names (event range not known until the end of the loop)
    gt.trun = TRun((out_dir_name / "trun.root").as_posix())
    gt.trunshowersim = TRunShowerSim((out_dir_name / "trunshowersim.root").as_posix())
    gt.trunefieldsim = TRunEfieldSim((out_dir_name / "trunefieldsim.root").as_posix())
    gt.tshower = TShower((out_dir_name / "tshower.root").as_posix())
    gt.tshowersim = TShowerSim((out_dir_name / "tshowersim.root").as_posix())
    gt.tefield = TEfield((out_dir_name / "tefield.root").as_posix())

    return out_dir_name


# Convert the RawShowerTree first entry to run values
def rawshower2grandrootrun(trawshower, gt):
    gt.trunshowersim.run_number = trawshower.run_number
    ## Name and version of the shower simulator
    gt.trunshowersim.sim_name = trawshower.sim_name

    #### ZHAireS/Coreas
    # * THINNING *
    # Thinning energy, relative to primary energy
    # this is EFRCTHN in Coreas (the 0th THIN value)
    gt.trunshowersim.rel_thin = trawshower.rel_thin

    # this is the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax
    # this is WMAX in Coreas (the 1st THIN value) - Weight limit for thinning
    gt.trunshowersim.maximum_weight = trawshower.maximum_weight

    # this is the ratio of energy at wich thining starts in hadrons and electromagnetic particles. In Aires is always 1
    # this is THINRAT in Coreas (the 0th THINH value) - hadrons
    gt.trunshowersim.hadronic_thinning = trawshower.hadronic_thinning

    # this is the ratio of electromagnetic to hadronic maximum weights.
    # this is WEIRAT in Coreas (the 1st THINH value)
    gt.trunshowersim.hadronic_thinning_weight = trawshower.hadronic_thinning_weight

    # Maximum radius (in cm) at observation level within which all particles are subject to inner radius thinning. In corsika particles are sampled following a r^(-4) distribution
    # Aires has a similar feature, but the definition is much more complex...so this will be left empty for now.
    # this is RMAX in Coreas (the 2nd THIN value)
    # gt.trunshowersim.rmax = trawshower.rmax

    # * CUTS *
    # gamma energy cut (GeV)
    gt.trunshowersim.lowe_cut_gamma = trawshower.lowe_cut_gamma

    # electron/positron energy cut (GeV)
    gt.trunshowersim.lowe_cut_e = trawshower.lowe_cut_e

    # muons energy cut (GeV)
    gt.trunshowersim.lowe_cut_mu = trawshower.lowe_cut_mu

    # mesons energy cut (GeV)
    gt.trunshowersim.lowe_cut_meson = trawshower.lowe_cut_meson

    # nucleons energy cut (GeV)
    gt.trunshowersim.lowe_cut_nucleon = trawshower.lowe_cut_nucleon


# Convert the RawEfieldTree first entry to run values
def rawefield2grandrootrun(trawefield, gt):
    gt.trunefieldsim.run_number = trawefield.run_number

    ## Name and version of the electric field simulator
    gt.trunefieldsim.efield_sim = trawefield.efield_sim

    ## Name of the atmospheric index of refraction model
    gt.trunefieldsim.refractivity_model = trawefield.refractivity_model
    gt.trunefieldsim.refractivity_model_parameters = trawefield.refractivity_model_parameters

    # The TRun run number
    gt.trun.run_number = trawefield.run_number

    ## The antenna time window is defined around a t0 that changes with the antenna, starts on t0+t_pre (thus t_pre is usually negative) and ends on t0+post
    gt.trunefieldsim.t_pre = trawefield.t_pre
    gt.trunefieldsim.t_post = trawefield.t_post


def get_tree_du_id_and_xyz(trawefield):
    # *** Store the DU's to run - they needed to be collected from all events ***
    # Get the ids and positions from all the events
    count = trawefield.draw("du_id:du_x:du_y:du_z", "", "goff")
    du_ids = np.array(np.frombuffer(trawefield.get_v1(), dtype=np.float64, count=count)).astype(int)
    du_xs = np.array(np.frombuffer(trawefield.get_v2(), dtype=np.float64, count=count)).astype(np.float32)
    du_ys = np.array(np.frombuffer(trawefield.get_v3(), dtype=np.float64, count=count)).astype(np.float32)
    du_zs = np.array(np.frombuffer(trawefield.get_v4(), dtype=np.float64, count=count)).astype(np.float32)

    # Get indices of the unique du_ids
    # ToDo: sort?
    unique_dus_idx = np.unique(du_ids, return_index=True)[1]
    # Leave only the unique du_ids
    du_ids = du_ids[unique_dus_idx]
    # Stack x/y/z together and leave only the ones for unique du_ids
    du_xyzs = np.column_stack([du_xs, du_ys, du_zs])[unique_dus_idx]

    return np.asarray(du_ids), np.asarray(du_xyzs)


# Convert the RawShowerTree entries
def rawshower2grandroot(trawshower, gt):
    ### Event name (the task name, can be usefull to track the original simulation)
    ## ToDo: not in TShowerSim - decide
    # gt.tshowersim.event_name = trawshower.event_name

    ## Run and event number
    gt.tshower.run_number = trawshower.run_number
    gt.tshower.event_number = trawshower.event_number
    gt.tshowersim.run_number = trawshower.run_number
    gt.tshowersim.event_number = trawshower.event_number

    ### Event Date  (used to define the atmosphere and/or the magnetic field)
    # ToDo: Shouldn't it be an epoch already in sims?
    gt.tshowersim.event_date = int(time.mktime(time.strptime(trawshower.event_date, "%Y-%m-%d")))

    ### Random seed
    gt.tshowersim.rnd_seed = trawshower.rnd_seed

    ### Energy in neutrinos generated in the shower (GeV). Useful for invisible energy computation
    # gt.tshower.energy_in_neutrinos = trawshower.energy_in_neutrinos

    ### Primary energy (GeV)
    # ToDo: it should be a scalar on sim side
    gt.tshower.energy_primary = trawshower.energy_primary[0]

    ### Shower azimuth (deg, CR convention)
    gt.tshower.azimuth = trawshower.azimuth

    ### Shower zenith  (deg, CR convention)
    gt.tshower.zenith = trawshower.zenith

    ### Primary particle type (PDG)
    # ToDo: it should be a scalar on sim side
    gt.tshower.primary_type = trawshower.primary_type[0]

    # Primary injection point [m] in Shower coordinates
    gt.tshowersim.primary_inj_point_shc = trawshower.primary_inj_point_shc

    ### Primary injection altitude [m] in Shower Coordinates
    gt.tshowersim.primary_inj_alt_shc = trawshower.primary_inj_alt_shc

    # primary injection direction in Shower Coordinates
    gt.tshowersim.primary_inj_dir_shc = trawshower.primary_inj_dir_shc

    ### Atmospheric model name TODO:standardize
    gt.tshower.atmos_model = trawshower.atmos_model

    # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    gt.tshower.atmos_model_param = trawshower.atmos_model_param

    # Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    gt.tshowersim.atmos_altitude = trawshower.atmos_altitude
    gt.tshowersim.atmos_density = trawshower.atmos_density
    gt.tshowersim.atmos_depth = trawshower.atmos_depth

    ### Magnetic field parameters: Inclination, Declination, Fmodulus.: In shower coordinates. Declination
    # The Earth's magnetic field, B, is described by its strength, Fmodulus = |B|; its inclination, I, defined
    # as the angle between the local horizontal plane and the field vector; and its declination, D, defined
    # as the angle between the horizontal component of B, H, and the geographical North (direction of
    # the local meridian). The angle I is positive when B points downwards and D is positive when H is
    # inclined towards the East.
    gt.tshower.magnetic_field = trawshower.magnetic_field

    ### Shower Xmax depth  (g/cm2 along the shower axis)
    gt.tshower.xmax_grams = trawshower.xmax_grams

    ### Shower Xmax position in shower coordinates [m]
    gt.tshower.xmax_pos_shc = trawshower.xmax_pos_shc

    ### Distance of Xmax  [m] to the ground
    # gt.tshower.xmax_distance = trawshower.xmax_distance

    ### Altitude of Xmax  [m]. Its important for the computation of the index of refraction at maximum, and of the cherenkov cone
    # gt.tshower.xmax_alt = trawshower.xmax_alt

    ### high energy hadronic model (and version) used TODO: standarize
    gt.tshowersim.hadronic_model = trawshower.hadronic_model

    ### low energy model (and version) used TODO: standarize
    gt.tshowersim.low_energy_model = trawshower.low_energy_model

    ### Time it took for the simulation of the cascade (s). In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation
    gt.tshowersim.cpu_time = trawshower.cpu_time

    ###META ZHAireS/Coreas

    ### Core position with respect to the antenna array (undefined for neutrinos)
    ## ToDo: conversion?
    gt.tshower.shower_core_pos = trawshower.shower_core_pos

    ### Longitudinal Pofiles (those compatible between Coreas/ZHAires)

    ## Longitudinal Profile of vertical depth (g/cm2) #we remove these becouse is not easily available in CORSIKA 
    #gt.tshowersim.long_depth = trawshower.long_depth
    ## Longitudinal Profile of slant depth (g/cm2)
    #gt.tshowersim.long_pd_depth = trawshower.long_slantdepth
    gt.tshowersim.long_pd_depth = trawshower.long_pd_depth
    ## Longitudinal Profile of Number of Gammas
    gt.tshowersim.long_pd_gammas = trawshower.long_pd_gammas
    ## Longitudinal Profile of Number of e+
    gt.tshowersim.long_pd_eplus = trawshower.long_pd_eplus
    ## Longitudinal Profile of Number of e-
    gt.tshowersim.long_pd_eminus = trawshower.long_pd_eminus
    ## Longitudinal Profile of Number of mu+
    gt.tshowersim.long_pd_muplus = trawshower.long_pd_muplus
    ## Longitudinal Profile of Number of mu-
    gt.tshowersim.long_pd_muminus = trawshower.long_pd_muminus
    ## Longitudinal Profile of Number of All charged particles
    gt.tshowersim.long_pd_allch = trawshower.long_pd_allch
    ## Longitudinal Profile of Number of Nuclei
    gt.tshowersim.long_pd_nuclei = trawshower.long_pd_nuclei
    ## Longitudinal Profile of Number of Hadrons
    gt.tshowersim.long_pd_hadr = trawshower.long_pd_hadr

    ## Longitudinal Profile of Energy of created neutrinos (GeV)
    gt.tshowersim.long_ed_neutrino = trawshower.long_ed_neutrino

    ## Longitudinal Profile of low energy gammas (GeV)
    gt.tshowersim.long_ed_gamma_cut = trawshower.long_ed_gamma_cut
    ## Longitudinal Profile of low energy e+/e- (GeV)
    gt.tshowersim.long_ed_e_cut = trawshower.long_ed_e_cut
    ## Longitudinal Profile of low energy mu+/mu- (GeV)
    gt.tshowersim.long_ed_mu_cut = trawshower.long_ed_mu_cut
    ## Longitudinal Profile of low energy hadrons (GeV)
    gt.tshowersim.long_ed_hadr_cut = trawshower.long_ed_hadr_cut

    ## Longitudinal Profile of energy deposit by gammas (GeV)
    gt.tshowersim.long_ed_gamma_ioniz = trawshower.long_ed_gamma_ioniz
    ## Longitudinal Profile of energy deposit by e+/e-  (GeV)
    gt.tshowersim.long_ed_e_ioniz = trawshower.long_ed_e_ioniz
    ## Longitudinal Profile of energy deposit by muons  (GeV)
    gt.tshowersim.long_ed_mu_ioniz = trawshower.long_ed_mu_ioniz
    ## Longitudinal Profile of energy deposit by hadrons (GeV)
    gt.tshowersim.long_ed_hadr_ioniz = trawshower.long_ed_hadr_ioniz

    # extra values
    gt.tshowersim.long_ed_depth = trawshower.long_ed_depth

    # gt.tshower.first_interaction = trawshower.first_interaction

# Convert the RawEfieldTree entries
def rawefield2grandroot(trawefield, gt):
    ## Run and event number
    gt.tefield.run_number = trawefield.run_number
    gt.tefield.event_number = trawefield.event_number

    gt.tshowersim.atmos_refractivity = trawefield.atmos_refractivity

    # Per antenna things
    gt.tefield.du_id = trawefield.du_id
    # gt.tefield.du_name = trawefield.du_name
    ## Number of detector units in the event - basically the antennas count
    gt.tefield.du_count = trawefield.du_count

    # ToDo!!!
    # gt.tefield.t_0 = trawefield.t_0
    gt.tefield.p2p = trawefield.p2p

    # ToDo: this should be a single vector of xyz
    ## X position in shower referential
    gt.tefield.du_x = trawefield.du_x
    ## Y position in shower referential
    gt.tefield.du_y = trawefield.du_y
    ## Z position in shower referential
    gt.tefield.du_z = trawefield.du_z

    ## Efield trace in X,Y,Z direction
    gt.tefield.trace = np.moveaxis(np.array([trawefield.trace_x, trawefield.trace_y, trawefield.trace_z]), 0,1)

    # Generate trigger times from t0s
    tempseconds=np.zeros((len(trawefield.t_0)), dtype=np.int64)
    tempseconds[:]=gt.tshowersim.event_seconds
    tempnanoseconds= np.int64(gt.tshowersim.event_nanoseconds + trawefield.t_0)
    #rolling over the nanoseconds    
    maskplus= gt.tshowersim.event_nanoseconds + trawefield.t_0 >=1e9
    maskminus= gt.tshowersim.event_nanoseconds + trawefield.t_0 <0
    tempnanoseconds[maskplus]-=np.int64(1e9)
    tempseconds[maskplus]+=np.int64(1)   
    tempnanoseconds[maskminus]+=np.int64(1e9)
    tempseconds[maskminus]-=np.int64(1)
    gt.tefield.du_nanoseconds=tempnanoseconds
    gt.tefield.du_seconds=tempseconds
    

# Convert the RawMetaTree entries
def rawmeta2grandroot(trawmeta, gt):
    gt.tshower.shower_core_pos = trawmeta.shower_core_pos
    gt.tshowersim.event_weight = trawmeta.event_weight
    gt.tshowersim.tested_cores = trawmeta.tested_cores
    #event time    
    if(trawmeta.unix_second>0):
      gt.tshower.core_time_s = trawmeta.unix_second              #this will be filled by the reconstruction of the core position eventually?
      gt.tshowersim.event_seconds = trawmeta.unix_second
    else:
      gt.tshower.core_time_s = 200854852
      gt.tshowersim.event_seconds = 200854852
    gt.tshower.core_time_ns = trawmeta.unix_nanosecond         #this will be filled by the reconstruction of the core position eventually?
    gt.tshowersim.event_nanoseconds = trawmeta.unix_nanosecond
    
    

## Get origin geoid
def get_origin_geoid(clargs):
    origin_geoid = [clargs.latitude, clargs.longitude, clargs.altitude]
    return origin_geoid

# Form the proper output directory name from command line arguments
def form_directory_name(clargs, date, time, run_number):
    # Change possible underscores in extra into -
    extra = clargs.extra.replace("_", "-")

    # Go through serial numbers in directory names to find a one that does not exist
    for sn in range(1000):
        dir_name = Path(clargs.output_parent_directory, f"sim_{clargs.site_name}_{date}_{time}_RUN{run_number:0>4}_CD_{extra}_{sn:0>4}")
        if not dir_name.exists():
            break
    # If directories with serial number up to 1000 already created
    else:
        print("All directories with serial number up to 1000 already exist. Please clean up some directories!")
        exit(0)

    return dir_name

# Rename the created files to appropriate names
def rename_files(clargs, path, start_event_number, end_event_number):
    # Go through output files
    for fn_start in ["trun", "trunshowersim", "trunefieldsim", "tshower", "tshowersim", "tefield"]:
        # Go through serial numbers in directory names to find a one that does not exist
        for sn in range(1000):
            fn_in = Path(path, f"{fn_start}.root")
            # Proper name of the file
            fn_out = Path(path, f"{fn_start}_{start_event_number}-{end_event_number}_L{clargs.analysis_level}_{sn:0>4}.root")
            # If the output file with the current serial number does not exist, rename to it
            if not fn_out.exists():
                fn_in.rename(fn_out)
                break
        else:
            print(f"Could not find a free filename for {fn_in} until serial number 1000. Please clean up some files!")
            exit(0)


if __name__ == '__main__':
    main()
