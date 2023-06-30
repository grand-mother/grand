import ROOT
import grand.io.root_trees as groot


class RootFile:
    # Use dict to associate rootfile ttree class to root_tree classe
    RunTrees = ["trun", "trunefieldsimdata","trunvoltage","trunefieldsim","trunshowersim","trunnoise"]
    EventTrees = ["teventefield", "teventshowersimdata",  "teventshower","teventvoltage",
                  "tadc","trawvoltage","tvoltage","tefield","tshower","tshowersim"]
    TreeToClass = {'trun': groot.TRun,
                   'teventadc': groot.TADC,
                   'teventvoltage': groot.TVoltage,
                   'teventefield': groot.TEfield,
                   'teventshower': groot.TShower,
                   'trunvoltagesimdata': groot.TRunVoltage,
                   'teventvoltagesimdata': groot.TVoltage,
                   'trunefieldsimdata': groot.TRunEfieldSim,
                   'teventefieldsimdata': groot.TEfield,
                   'trunsimdata': groot.TRunEfieldSim,
                   'teventshowersimdata': groot.TShower,
                   'teventshowerzhaires': groot.TShowerSim,
                   'tdetectorinfo': groot.TRun,
                   'trunvoltage':  groot.TRunVoltage,
                   'tadc':  groot.TADC,
                   'trawvoltage':  groot.TRawVoltage,
                   'tvoltage':  groot.TVoltage,
                   'tefield':  groot.TEfield,
                   'tshower':  groot.TShower,
                   'trunefieldsim':  groot.TRunEfieldSim,
                   'trunshowersim':  groot.TRunShowerSim,
                   'tshowersim':  groot.TShowerSim,
                   'trunnoise': groot.TRunNoise
                   }
    tefieldToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'time_seconds': 'efield_time_seconds',
        'time_nanoseconds': 'efield_time_nanoseconds',
        'event_type': 'efield_id_event_type',
        'du_count': 'efield_du_count',
        'du_id': 'du_id'
    }
    tshowersimToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'input_name': 'sim_input_name',
        'event_date': 'sim_event_date',
        'rnd_seed': 'sim_rnd_seed',
        'sim_primary_energy': 'sim_primary_energy',
        'sim_primary_type': 'sim_id_primary_type',
        'sim_primary_inj_point_shc': 'sim_primary_inj_point_shc',
        'sim_primary_inj_alt_shc': 'sim_primary_inj_alt_shc',
        'sim_primary_inj_dir_shc': 'sim_primary_inj_dir_shc',
        'hadronic_model': 'sim_id_hadronic_model',
        'low_energy_model': 'sim_id_low_energy_model',
        'cpu_time': 'sim_cpu_time',
        'long_depth': 'sim_long_depth',
        'long_eminus': 'sim_long_eminus',
        'long_eplus': 'sim_long_eplus',
        'long_muminus': 'sim_long_muminus',
        'long_muplus': 'sim_long_muplus',
        'long_gammas': 'sim_long_gammas',
        'long_hadrons': 'sim_long_hadrons',
        'long_gamma_elow': 'sim_long_gamma_elow',
        'long_e_elow': 'sim_long_e_elow',
        'long_e_edep': 'sim_long_e_edep',
        'long_mu_edep': 'sim_long_mu_edep',
        'long_mu_elow': 'sim_long_mu_elow',
        'long_hadron_edep': 'sim_long_hadron_edep',
        'long_hadron_elow': 'sim_long_hadron_elow',
        'long_neutrinos': 'sim_long_neutrinos'
    }
    trunshowersimToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'rel_thin': 'rel_thin',
        'weight_factor': 'weight_factor',
        'lowe_cut_e': 'lowe_cut_e',
        'lowe_cut_gamma': 'lowe_cut_gamma',
        'lowe_cut_mu': 'lowe_cut_mu',
        'lowe_cut_meson': 'lowe_cut_meson',
        'lowe_cut_nucleon': 'lowe_cut_nucleon',
        'site': 'id_site',
        'sim_name': 'sim_name',
        'sim_version': 'sim_version'
    }
    trunefieldsimToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'refractivity_model': 'id_refractivity_model',
        'refractivity_model_parameters': 'refractivity_model_parameters',
        't_pre': 't_pre',
        't_post': 't_post',
        't_bin_size': 't_bin_size',
        'sim_name': 'efield_sim_name',
        'sim_version': 'efield_sim_version'
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
#        'site_layout': 'id_site_layout',
        'origin_geoid': 'origin_geoid',
        't_bin_size': 't_bin_size'
    }

    trunnoiseToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'GalNoiseMap': 'GalNoiseMap',
        'GalNoiseLST': 'GalNoiseLST'
    }
    tadcToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'event_size': 'adc_event_size',
        't3_number': 'adc_t3_number',
        'first_du': 'adc_id_first_du',
        'time_seconds': 'adc_time_seconds',
        'time_nanoseconds': 'adc_time_nanoseconds',
        'event_type': 'adc_id_event_type',
        'event_version': 'adc_event_version',
        'du_count': 'adc_du_count',
        'du_id': 'du_id'
    }
    trawvoltageToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'first_du': 'voltage_id_first_du',
        'time_seconds': 'voltage_time_seconds',
        'time_nanoseconds': 'voltage_time_nanoseconds',
        'du_count': 'voltage_du_count'
    }
    tvoltageToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'first_du': 'voltage_id_first_du',
        'time_seconds': 'voltage_time_seconds',
        'time_nanoseconds': 'voltage_time_nanoseconds',
        'du_count': 'voltage_du_count'
    }
    tshowerToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'primary_type': 'id_primary_type',
        'energy_em': 'energy_em',
        'energy_primary': 'energy_primary',
        'azimuth': 'azimuth',
        'zenith': 'zenith',
        'direction': 'direction',
        'shower_core_pos': 'shower_core_pos',
        'atmos_model': 'id_atmos_model',
        'atmos_model_param': 'atmos_model_param',
        'magnetic_field': 'magnetic_field',
        'core_alt': 'core_alt',
        'xmax_grams': 'xmax_grams',
        'xmax_pos': 'xmax_pos',
        'xmax_pos_shc': 'xmax_pos_shc',
        'core_time_s': 'core_time_s',
        'core_time_ns': 'core_time_ns'
    }

#### ---- These are conversions for previous version of root files ---- ####
###  ---- adapted to the new structure of the DB                   ---- ####
###  ---- Keeped for compatibility but will be removed in the future -- ####


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
        'run_number': 'run_number',
        'event_number': 'event_number',
        'event_name': 'sim_input_name',
        'date': 'sim_event_date',
        'rnd_seed': 'sim_rnd_seed',
        #'energy_in_neutrinos': 'sim_energy_in_neutrinos',
        'prim_energy': 'sim_primary_energy',
        #'shower_azimuth': 'sim_shower_azimuth',
        #'shower_zenith': 'sim_shower_zenith',
        'prim_type': 'sim_id_primary_type',
        'prim_injpoint_shc': 'sim_primary_inj_point_shc',
        'prim_inj_alt_shc': 'sim_primary_inj_alt_shc',
        'prim_inj_dir_shc': 'sim_primary_inj_dir_shc',
        #'atmos_model': 'sim_id_atmos_model',
        #'atmos_model_param': 'sim_atmos_model_param',
        #'magnetic_field': 'sim_magnetic_field',
        #'xmax_grams': 'sim_xmax_grams',
        #'xmax_pos_shc': 'sim_xmax_pos_shc',
        #'xmax_distance': 'sim_xmax_distance',
        #'xmax_alt': 'sim_xmax_alt',
        'hadronic_model': 'sim_id_hadronic_model',
        'low_energy_model': 'sim_id_low_energy_model',
        #'cpu_time': 'sim_cpu_time' # Problem of type !!!!
    }
    teventshowerzhairesToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'relative_thining': 'rel_thin',
        'weight_factor': 'weight_factor',
        'gamma_energy_cut': 'lowe_cut_gamma',
        'electron_energy_cut': 'lowe_cut_e',
        'muon_energy_cut': 'lowe_cut_mu',
        'meson_energy_cut': 'lowe_cut_meson',
        'nucleon_energy_cut': 'lowe_cut_nucleon',
#        'other_parameters': 'zhaires_other_parameters'
    }
    trunefieldsimdataToDB = {
        'table': 'run',
        'run_number': 'run_number',
        'refractivity_model': 'id_refractivity_model',
        'refractivity_model_parameters': 'refractivity_model_parameters',
        't_pre': 't_pre',
        't_post': 't_post',
        't_bin_size': 't_bin_size'
    }

# trunToDB (same name in old and new versions) are
# almost identical in both versions... thus new version works also with old files

    teventvoltageToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'event_size': 'adc_event_size',
        't3_number': 'adc_t3_number',
        'first_du': 'adc_id_first_du',
        'time_seconds': 'adc_time_seconds',
        'time_nanoseconds': 'adc_time_nanoseconds',
        'event_type': 'adc_id_event_type',
        'event_version': 'adc_event_version',
        'du_count': 'adc_du_count',
        'du_id': 'du_id'
    }
    teventshowerToDB = {
        'table': 'event',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'shower_type': 'id_primary_type',
        'shower_energy': 'energy_primary',
        'shower_azimuth': 'azimuth',
        'shower_zenith': 'zenith',
        'shower_core_pos': 'shower_core_pos',
        'atmos_model': 'id_atmos_model',
        'atmos_model_param': 'atmos_model_param',
        'magnetic_field': 'magnetic_field',
        #'date': 'event_date',
        'ground_alt': 'core_alt',
        'xmax_grams': 'xmax_grams',
        'xmax_pos_shc': 'xmax_pos_shc',
        #'xmax_alt': 'xmax_alt',
        #'gh_fit_param': 'gh_fit_param',
        'core_time': 'core_time_s'
    }

#### ------------------------------------------------------------------ ####


    TreeList = {}

    ## We retreive the list of Ttrees in the file  and store them as the corresponding class from root_files.py in the dict TreeList
    def __init__(self, f_name):
        myfile = ROOT.TFile(f_name)

        for key in myfile.GetListOfKeys():
            name = key.GetName()
            #Names of trees should start with their type followed by _ and whatever
            ttype = name.split('_', 1)[0]

            if ttype in self.TreeToClass:
                self.TreeList[ttype] = self.TreeToClass[ttype](f_name)
            else:
                print(ttype + " is unknown")

    def copy_content_to(self, file):
        for treename in self.TreeList:
            tree = self.TreeToClass[treename](file)
            tree.copy_contents(self.TreeList[treename])
            tree.write()

