import ROOT
import grand.dataio.root_trees as groot
import grand.manage_log as mlg
import os
logger = mlg.get_logger_for_script(__name__)
#mlg.create_output_for_logger("debug", log_stdout=True)

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

    metaToDB = {
        '_type' : 'id_tree_type',
        '_tree_name' : 'tree_name',
        '_comment' : 'comment',
        '_creation_datetime' : 'creation_datetime',
        '_modification_software' : 'id_modification_software',
#        '_modification_software_version' : 'id_modification_software_version',
        '_source_datetime' : 'source_datetime',
        '_analysis_level' : 'analysis_level',
        '_modification_history' : 'modification_history'
    }

    trunToDB = {
        'table': 'trun',
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
        'site_layout': 'id_site_layout',
        'origin_geoid': 'origin_geoid',
        #        't_bin_size': 't_bin_size',
        'du_id': 'du_id'
    }
    trunshowersimToDB = {
        'table': 'trunshowersim',
        'run_number': 'run_number',
        'rel_thin': 'rel_thin',
        'lowe_cut_e': 'lowe_cut_e',
        'lowe_cut_gamma': 'lowe_cut_gamma',
        'lowe_cut_mu': 'lowe_cut_mu',
        'lowe_cut_meson': 'lowe_cut_meson',
        'lowe_cut_nucleon': 'lowe_cut_nucleon',
        'site': 'id_site',
        'sim_name': 'sim_name',
        'sim_version': 'sim_version',
        'maximum_weight': 'maximum_weight',
        'hadronic_thinning': 'hadronic_thinning',
        'hadronic_thinning_weight': 'hadronic_thinning_weight'
    }
    trunefieldsimToDB = {
        'table': 'trunefieldsim',
        'run_number': 'run_number',
        'refractivity_model': 'id_refractivity_model',
        'refractivity_model_parameters': 'refractivity_model_parameters',
        't_pre': 't_pre',
        't_post': 't_post',
        'sim_name': 'sim_name',
        'sim_version': 'sim_version'
    }
    trunvoltageToDB = {
        'table': 'trunvoltage',
        'run_number': 'run_number',
        'firmware_version': 'firmware_version',
        'adc_sampling_frequency': 'adc_sampling_frequency',
        'adc_sampling_resolution': 'adc_sampling_resolution',
        'adc_input_channels': 'adc_input_channels',
        'adc_enabled_channels': 'adc_enabled_channels'
    }
    trunnoiseToDB = {
        'table': 'trunnoise',
        'run_number': 'run_number',
        'GalNoiseMap': 'galnoisemap',
        'GalNoiseLST': 'galnoiselst'
    }
    trunefieldsimdataToDB = {
        'table': 'trunefieldsim',
        'run_number': 'run_number',
        'refractivity_model': 'id_refractivity_model',
        'refractivity_model_parameters': 'refractivity_model_parameters',
        't_pre': 't_pre',
        't_post': 't_post',
        #        't_bin_size': 't_bin_size'
    }
    tefieldToDB = {
        'table': 'events',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'time_seconds': 'time_seconds',
        'time_nanoseconds': 'time_nanoseconds',
        'event_type': 'event_type',
        'du_count': 'du_count',
        'du_id': 'du_id'
    }
    tshowersimToDB = {
        'table': 'tshowersim',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'input_name': 'input_name',
        'event_date': 'event_date',
        'rnd_seed': 'rnd_seed',
        'primary_inj_point_shc': 'primary_inj_point_shc',
        'primary_inj_alt_shc': 'primary_inj_alt_shc',
        'primary_inj_dir_shc': 'primary_inj_dir_shc',
        'hadronic_model': 'id_hadronic_model',
        'low_energy_model': 'id_low_energy_model',
        'cpu_time': 'cpu_time',
        #'long_pd_depth': 'long_pd_depth',
        #'long_pd_eminus': 'long_pd_eminus',
        #'long_pd_eplus': 'long_pd_eplus',
        #'long_pd_muminus': 'long_pd_muminus',
        #'long_pd_muplus': 'long_pd_muplus',
        #'long_pd_gamma': 'long_pd_gamma',
        #'long_pd_hadron': 'long_pd_hadron',
        #'long_gamma_elow': 'long_gamma_elow',
        #'long_e_elow': 'long_e_elow',
        #'long_e_edep': 'long_e_edep',
        #'long_mu_edep': 'long_mu_edep',
        #'long_mu_elow': 'long_mu_elow',
        #'long_hadron_edep': 'long_hadron_edep',
        #'long_hadron_elow': 'long_hadron_elow',
        #'long_neutrino': 'long_neutrino',
        'event_weight': 'event_weight'
    }
    tadcToDB = {
        'table': 'events',
        'run_number': 'run_number',
        'event_number': 'event_number',
#        'event_size': 'event_size',
#        't3_number': 't3_number',
        'first_du': 'first_du',
        'time_seconds': 'time_seconds',
        'time_nanoseconds': 'time_nanoseconds',
        'event_type': 'event_type',
#        'event_version': 'event_version',
        'du_count': 'du_count',
        'du_id': 'du_id'
    }
    trawvoltageToDB = {
        'table': 'events',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'first_du': 'first_du',
        'time_seconds': 'time_seconds',
        'time_nanoseconds': 'time_nanoseconds',
        'du_count': 'du_count',
        'du_id': 'du_id'
    }
    tvoltageToDB = {
#        'table': 'tvoltage',
#        'run_number': 'run_number',
#        'event_number': 'event_number',
#        'first_du': 'first_du',
#        'time_seconds': 'time_seconds',
#        'time_nanoseconds': 'time_nanoseconds',
#        'du_count': 'du_count'
    }
    tshowerToDB = {
        'table': 'tshower',
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


#    teventefieldToDB = {
#        'table': 'tefield',
#        'run_number': 'run_number',
#        'event_number': 'event_number',
#        'time_seconds': 'efield_time_seconds',
#        'time_nanoseconds': 'efield_time_nanoseconds',
#        'event_type': 'id_event_type',
#        'du_count': 'du_count',
#        'du_id': 'du_id'
#    }
    teventshowersimdataToDB = {
        'table': 'tshowersim',
        'run_number': 'run_number',
        'event_number': 'event_number',
        'event_name': 'input_name',
        'date': 'event_date',
        'rnd_seed': 'rnd_seed',
        #'energy_in_neutrinos': 'energy_in_neutrinos',
        'prim_energy': 'primary_energy',
        #'shower_azimuth': 'shower_azimuth',
        #'shower_zenith': 'shower_zenith',
        'prim_type': 'id_primary_type',
        'prim_injpoint_shc': 'primary_inj_point_shc',
        'prim_inj_alt_shc': 'primary_inj_alt_shc',
        'prim_inj_dir_shc': 'primary_inj_dir_shc',
        #'atmos_model': 'id_atmos_model',
        #'atmos_model_param': 'atmos_model_param',
        #'magnetic_field': 'magnetic_field',
        #'xmax_grams': 'xmax_grams',
        #'xmax_pos_shc': 'xmax_pos_shc',
        #'xmax_distance': 'xmax_distance',
        #'xmax_alt': 'xmax_alt',
        'hadronic_model': 'id_hadronic_model',
        'low_energy_model': 'id_low_energy_model',
        #'cpu_time': 'cpu_time' # Problem of type !!!!
    }
    teventshowerzhairesToDB = {
        'table': 'trunshowersim',
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


# trunToDB (same name in old and new versions) are
# almost identical in both versions... thus new version works also with old files

#    teventvoltageToDB = {
#        'table': 'tadc',
#        'run_number': 'run_number',
#        'event_number': 'event_number',
#        'event_size': 'event_size',
#        't3_number': 't3_number',
#        'first_du': 'first_du',
#        'time_seconds': 'time_seconds',
#        'time_nanoseconds': 'time_nanoseconds',
#        'event_type': 'id_event_type',
#        'event_version': 'event_version',
#        'du_count': 'du_count',
##        'du_id': 'du_id'
#    }
    teventshowerToDB = {
        'table': 'tshower',
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

    #TreeList is a dict with name of the trees as key and the class corresponding to it's type as value
    TreeList = {}
    file = None
    filename = None

    ## We retreive the list of Ttrees in the file  and store them as the corresponding class from root_files.py in the dict TreeList
    def __init__(self, f_name):
        self.filename = f_name
        self.TreeList.clear()
        self.file = ROOT.TFile(f_name)
        for key in self.file.GetListOfKeys():
            tname = key.GetName()
            #Names of trees should start with their type followed by _ and whatever
            # so we extract the type of the tree to get the correct class
            ttype = tname.split('_', 1)[0]

            if ttype in self.TreeToClass:
                self.TreeList[tname] = self.TreeToClass[ttype](f_name)
            else:
                logger.warning(f"{ttype} is unknown")




    def copy_content_to(self, file):
        for treename in self.TreeList:
            tree = self.TreeToClass[treename](file)
            tree.copy_contents(self.TreeList[treename])
            tree.write()

    ## Function to determine the dataset name for a file.
    # dataset naming convention is : [sim|exp|mod]_[site]_[date]_[time]_[extra]_[serial]
    # For observations :
    # [sim|exp|mod] -> given by data_source in Trun (detector->exp)
    # [site] -> given by site in Trun
    # [date]_[time] -> given by first_event_time in Trun
    # [extra]-> given by user (metadata ?)
    # serial -> automatically incremented in case of new version (how to do that ?)
    def dataset_name(self):
        name = os.path.basename(os.path.dirname(self.filename))
        return name
        treename = 'trun'
        name = "noname"
        for run in self.TreeList[treename].get_list_of_runs():
            self.TreeList[treename].get_run(run)
            source = self.TreeList[treename].data_source
            if source == "detector":
                source = "det"
            source = sanitize_name(source)
            site = sanitize_name(self.TreeList[treename].site)
            mydate, mytime = timestamp_to_date_time(self.TreeList[treename].first_event_time)
            extra = ""
            serial = "1"
            name = source+"_"+site+"_"+mydate+"_"+mytime+"_"+extra+"_"+serial
            #print(name)
            #We use only first run
            break
        return name

    # Remove any underscore or space
def sanitize_name(name):
    import re
    return re.sub(r"\s+|_+", '', name)

def timestamp_to_date_time(timestamp):
    import time
    return time.strftime("%Y%m%d", time.gmtime(float(timestamp))), time.strftime("%H%M%S", time.gmtime(float(timestamp)))