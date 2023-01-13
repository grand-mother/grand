#import uproot
import ROOT
import grand.io.root_trees as groot


class RootFile:
    # Use dict to associate rootfile ttree class to root_tree classe
    TreeToClass = {'trun': groot.RunTree,
                   'teventadc': groot.ADCEventTree,
                   'teventvoltage': groot.VoltageEventTree,
                   'teventefield': groot.EfieldEventTree,
                   'teventshower': groot.ShowerEventTree,
                   'trunvoltagesimdata': groot.VoltageRunSimdataTree,
                   'teventvoltagesimdata': groot.VoltageEventSimdataTree,
                   'trunefieldsimdata': groot.EfieldRunSimdataTree,
                   'teventefieldsimdata': groot.EfieldEventSimdataTree,
                   'trunsimdata': groot.ShowerRunSimdataTree,
                   'teventshowersimdata': groot.ShowerEventSimdataTree,
                   'teventshowerzhaires': groot.ShowerEventZHAireSTree,
                   'tdetectorinfo': groot.DetectorInfo
                   }
    teventefieldToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'time_seconds': 'efield_time_seconds',
        'time_nanoseconds': 'efield_time_nanoseconds',
        'event_type': 'efield_id_event_type',
        'du_count': 'efield_du_count',
        'du_id': 'du_id'
    }
    teventshowersimdataToDB = {
        'table': 'event',
        'event_name': 'sim_event_name',
        'date': 'sim_event_date',
        'rnd_seed': 'sim_rnd_seed',
        'energy_in_neutrinos': 'sim_energy_in_neutrinos',
        'prim_energy': 'sim_prim_energy',
        'shower_azimuth': 'sim_shower_azimuth',
        'shower_zenith': 'sim_shower_zenith',
        'prim_type': 'sim_id_prim_type',
        'prim_injpoint_shc': 'sim_prim_injpoint_shc',
        'prim_inj_alt_shc': 'sim_prim_inj_alt_shc',
        'prim_inj_dir_shc': 'sim_prim_inj_dir_shc',
        'atmos_model': 'sim_id_atmos_model',
        'atmos_model_param': 'sim_atmos_model_param',
        'magnetic_field': 'sim_magnetic_field',
        'xmax_grams': 'sim_xmax_grams',
        'xmax_pos_shc': 'sim_xmax_pos_shc',
        'xmax_distance': 'sim_xmax_distance',
        'xmax_alt': 'sim_xmax_alt',
        'hadronic_model': 'sim_id_hadronic_model',
        'low_energy_model': 'sim_id_low_energy_model',
#        'cpu_time': 'sim_cpu_time'
    }
    teventshowerzhairesToDB = {
        'table': 'event',
        'relative_thining': 'zhaires_rel_thining',
        'weight_factor': 'zhaires_weight_factor',
        'gamma_energy_cut': 'zhaires_gamma_energy_cut',
        'electron_energy_cut': 'zhaires_electron_energy_cut',
        'muon_energy_cut': 'zhaires_muon_energy_cut',
        'meson_energy_cut': 'zhaires_meson_energy_cut',
        'nucleon_energy_cut': 'zhaires_nucleon_energy_cut',
        'other_parameters': 'zhaires_other_parameters'
    }
    trunefieldsimdataToDB = {
        'table': 'run',
        'refractivity_model': 'id_refractivity_model',
        'refractivity_model_parameters': 'refractivity_model_parameters',
        't_pre': 't_pre',
        't_post': 't_post',
        't_bin_size': 't_bin_size'
    }
    trunToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'run_mode': 'id_run_mode',
        'first_event': 'first_event',
        'first_event_time': 'first_event_time',
        'last_event': 'last_event',
        'last_event_time': 'last_event_time',
        'data_source': 'id_data_source',
        'data_generator': 'id_data_generator',
        'data_generator_version': 'id_data_generator_version',
        'site': 'id_site',
        'site_long': 'site_long',
        'site_lat': 'site_lat',
        'origin_geoid': 'origin_geoid'
    }
    teventvoltageToDB = {
        'table': 'event',
        'event_size': 'voltage_event_size',
        't3_number': 'voltage_t3_number',
        'first_du': 'voltage_id_first_du',
        'time_seconds': 'voltage_time_seconds',
        'time_nanoseconds': 'voltage_time_nanoseconds',
        'event_type': 'voltage_id_event_type',
        'event_version': 'voltage_event_version',
        'du_count': 'voltage_du_count',
        'du_id': 'du_id'
    }
    teventshowerToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'shower_type': 'id_shower_type',
        'shower_energy': 'shower_energy',
        'shower_azimuth': 'shower_azimuth',
        'shower_zenith': 'shower_zenith',
        'shower_core_pos': 'shower_core_pos',
        'atmos_model': 'id_atmos_model',
        'atmos_model_param': 'atmos_model_param',
        'magnetic_field': 'magnetic_field',
        'date': 'event_date',
        'ground_alt': 'ground_alt',
        'xmax_grams': 'xmax_grams',
        'xmax_pos_shc': 'xmax_pos_shc',
        'xmax_alt': 'xmax_alt',
        'gh_fit_param': 'gh_fit_param',
        'core_time': 'core_time'
    }

    TreeList = {}

    ## We retreive the list of Ttrees in the file  and store them as the corresponding class from root_files.py in the dict TreeList
    def __init__(self, f_name):
        myfile = ROOT.TFile(f_name)
        for keyo in myfile.GetListOfKeys():
            key = keyo.GetName()
            if key in self.TreeToClass:
                self.TreeList[key] = self.TreeToClass[key](f_name)
            else:
                print(key + " is unknown")

    ## OLD VERSION WITH UPROOT
#    def __init__(self, f_name):
#        myfile = uproot.open(f_name)
#        for key in myfile.keys(cycle=False):
#            if key in self.TreeToClass:
#                self.TreeList[key] = self.TreeToClass[key](f_name)
#            else:
#                print(key + " is unknown")