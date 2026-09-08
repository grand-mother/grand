"""
Unit tests for the grand.dataio.root_trees module.

Jun 19, 2023, RK.
"""

import unittest
from tests import TestCase
from pathlib import Path
import os

import grand.dataio as groot
from grand import grand_get_path_root_pkg


class RootTreesTest(TestCase):
    """Unit tests for the root_files module"""

    def test_stdvectorlist(self):
        svl = groot.StdVectorList("float", [1.,2.,3.])
        self.assertTrue(isinstance(svl.vec_type, str))
        self.assertTrue(svl[0]==1.)

    def test_stdstring(self):
        ss = groot.StdString('GRAND')
        self.assertTrue(ss.string=='GRAND')

    def test_datatree(self):
        efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
        self.assertTrue((efield_file).exists())
        fi = groot.DataTree(str(efield_file))
        self.assertTrue(fi._file is not None)

    def test_motherruntree(self):
        fi = groot.MotherRunTree("test.root")
        self.assertTrue(len(fi.get_list_of_runs())==0)
        os.remove("test.root")

    def test_trun(self):
        self.trun = groot.TRun()
        self.assertTrue(self.trun._file is None)
        self.assertTrue(hasattr(self.trun, 'run_mode'))
        self.assertTrue(hasattr(self.trun, 'first_event'))
        self.assertTrue(hasattr(self.trun, 'first_event_time'))
        self.assertTrue(hasattr(self.trun, 'last_event'))
        self.assertTrue(hasattr(self.trun, 'last_event_time'))
        self.assertTrue(hasattr(self.trun, 'data_source'))
        self.assertTrue(hasattr(self.trun, 'data_generator'))
        self.assertTrue(hasattr(self.trun, 'data_generator_version'))
        self.assertTrue(hasattr(self.trun, 'event_type'))
        self.assertTrue(hasattr(self.trun, 'event_version'))
        self.assertTrue(hasattr(self.trun, 'site'))
        self.assertTrue(hasattr(self.trun, 'site_layout'))
        self.assertTrue(hasattr(self.trun, 'origin_geoid'))
        self.assertTrue(hasattr(self.trun, 'du_id'))
        self.assertTrue(hasattr(self.trun, 'du_geoid'))
        self.assertTrue(hasattr(self.trun, 'du_xyz'))
        self.assertTrue(hasattr(self.trun, 'du_type'))
        self.assertTrue(hasattr(self.trun, 'du_tilt'))
        self.assertTrue(hasattr(self.trun, 'du_ground_tilt'))
        self.assertTrue(hasattr(self.trun, 'du_nut'))
        self.assertTrue(hasattr(self.trun, 'du_feb'))
        self.assertTrue(hasattr(self.trun, 't_bin_size'))
        self.trun.run_mode = 1
        #self.assertTrue(isinstance(self.trun.run_mode, np.uint32))
        self.trun.first_event=1
        self.trun.first_event_time=1
        self.trun.last_event=1
        self.trun.last_event_time=1
        self.trun.data_source="Dunhuang"
        self.trun.data_generator="dontknow"
        self.trun.data_generator_version="1"
        self.trun.event_type=1
        self.trun.event_version=1
        self.trun.site="Dunhuang"
        self.trun.site_layout="GP300"
        self.trun.origin_geoid=[1,2,3] # not a real location
        self.trun.du_id=[1,2,3]
        self.trun.du_geoid=[[1,2,3],[1,2,3],[1,2,3]]
        self.trun.du_xyz=[[1,2,3],[1,2,3],[1,2,3]]
        self.trun.du_type=['a','a','a']
        self.trun.du_nut=[1,2,3]
        self.trun.du_feb=[1,2,3]
        self.trun.t_bin_size=[1.,2.,3.]

    # Removed 2026-09-08: test_trunvoltage, test_tadc, test_trunshowersim and
    # test_tshowersim.  Between them they made 110 hasattr checks that a field
    # exists on a tree.  73c959d switched all four off in January because the
    # schema had drifted under them, with a note that they needed updating;
    # they were never updated, and they still fail if re-enabled.
    #
    # 31 of those 110 names no longer exist on their class, renamed or dropped
    # -- adc_input_channels is adc_input_channels_ch now, and so on for the
    # other 30.  Rewriting the lists would just produce a fresh hand-maintained
    # copy of the names that happen to exist today, which is the thing that
    # rotted in the first place.
    #
    # What covers these classes instead is generated from the classes:
    #
    #   tests/dataio/schema_snapshot.json     pins all 304 fields of all 11
    #                                         tree classes, regenerated with
    #                                         --write rather than typed out
    #   tests/dataio/test_tree_roundtrip.py   writes and reads back every one
    #                                         of the 276 branch fields
    #
    # That covers all four -- TRunVoltage 18 fields, TADC 111, TRunShowerSim
    # 14, TShowerSim 34 -- and checks the values survive a write, which hasattr
    # never did.

    def test_trawvoltage(self):
        self.trvolt = groot.TRawVoltage()
        self.assertTrue(self.trvolt._file is None)
        self.assertTrue(hasattr(self.trvolt, 'event_size'))
        self.assertTrue(hasattr(self.trvolt, 'first_du'))
        self.assertTrue(hasattr(self.trvolt, 'time_seconds'))
        self.assertTrue(hasattr(self.trvolt, 'time_nanoseconds'))
        self.assertTrue(hasattr(self.trvolt, 'du_count'))
        self.assertTrue(hasattr(self.trvolt, 'du_id'))
        self.assertTrue(hasattr(self.trvolt, 'du_seconds'))        
        self.assertTrue(hasattr(self.trvolt, 'du_nanoseconds'))
        #self.assertTrue(hasattr(self.trvolt, 'trigger_position'))
        self.assertTrue(hasattr(self.trvolt, 'trigger_flag'))
        self.assertTrue(hasattr(self.trvolt, 'atm_temperature'))
        self.assertTrue(hasattr(self.trvolt, 'atm_pressure'))
        self.assertTrue(hasattr(self.trvolt, 'atm_humidity'))
        self.assertTrue(hasattr(self.trvolt, 'du_acceleration'))
        self.assertTrue(hasattr(self.trvolt, 'battery_level'))
        self.assertTrue(hasattr(self.trvolt, 'adc_samples_count_channel'))
        self.assertTrue(hasattr(self.trvolt, 'trigger_pattern'))
        self.assertTrue(hasattr(self.trvolt, 'trigger_rate'))
        self.assertTrue(hasattr(self.trvolt, 'clock_tick'))
        self.assertTrue(hasattr(self.trvolt, 'clock_ticks_per_second'))
        self.assertTrue(hasattr(self.trvolt, 'gps_offset'))
        self.assertTrue(hasattr(self.trvolt, 'gps_leap_second'))
        self.assertTrue(hasattr(self.trvolt, 'gps_status'))
        self.assertTrue(hasattr(self.trvolt, 'gps_alarms'))
        self.assertTrue(hasattr(self.trvolt, 'gps_warnings'))
        self.assertTrue(hasattr(self.trvolt, 'gps_time'))
        self.assertTrue(hasattr(self.trvolt, 'gps_long'))
        self.assertTrue(hasattr(self.trvolt, 'gps_lat'))
        self.assertTrue(hasattr(self.trvolt, 'gps_alt'))
        self.assertTrue(hasattr(self.trvolt, 'gps_temp'))
        self.assertTrue(hasattr(self.trvolt, 'ioff'))
        self.assertTrue(hasattr(self.trvolt, 'trace_ch'))
        self.trvolt.event_size=1
        self.trvolt.first_du=1
        self.trvolt.time_seconds=1
        self.trvolt.time_nanoseconds=1
        self.trvolt.du_count=1
        self.trvolt.du_id=[1]
        self.trvolt.du_seconds=[1]
        self.trvolt.du_nanoseconds=[1]
        self.trvolt.trigger_flag=[1]
        self.trvolt.atm_temperature=[1]
        self.trvolt.atm_pressure=[1]
        self.trvolt.atm_humidity=[1]
        self.trvolt.du_acceleration=[[1,2,3]]
        self.trvolt.battery_level=[1]
        self.trvolt.adc_samples_count_channel=[[1,2,3]]
        self.trvolt.trigger_pattern=[1]
        self.trvolt.trigger_rate=[1]
        self.trvolt.clock_tick=[1]
        self.trvolt.clock_ticks_per_second=[1]
        self.trvolt.gps_offset=[1]
        self.trvolt.gps_leap_second=[1]
        self.trvolt.gps_status=[1]
        self.trvolt.gps_alarms=[1]
        self.trvolt.gps_warnings=[1]
        self.trvolt.gps_time=[1]
        self.trvolt.gps_long=[1]
        self.trvolt.gps_lat=[1]
        self.trvolt.gps_alt=[1]
        self.trvolt.gps_temp=[1]
        self.trvolt.ioff=[1]
        self.trvolt.trace_ch=[[[1,2,3]]]

    def test_tvoltage(self):
        self.tvolt = groot.TVoltage()
        self.assertTrue(self.tvolt._file is None)
        self.assertTrue(hasattr(self.tvolt, 'first_du'))
        self.assertTrue(hasattr(self.tvolt, 'time_seconds'))
        self.assertTrue(hasattr(self.tvolt, 'time_nanoseconds'))
        self.assertTrue(hasattr(self.tvolt, 'du_count'))
        self.assertTrue(hasattr(self.tvolt, 'du_id'))
        self.assertTrue(hasattr(self.tvolt, 'du_seconds'))
        self.assertTrue(hasattr(self.tvolt, 'du_nanoseconds'))
        self.assertTrue(hasattr(self.tvolt, 'trigger_flag'))
        self.assertTrue(hasattr(self.tvolt, 'du_acceleration'))
        self.assertTrue(hasattr(self.tvolt, 'trigger_rate'))
        self.assertTrue(hasattr(self.tvolt, 'trace'))
        self.assertTrue(hasattr(self.tvolt, 'p2p'))
        self.assertTrue(hasattr(self.tvolt, 'time_max'))
        self.tvolt.first_du=1
        self.tvolt.time_seconds=1
        self.tvolt.time_nanoseconds=1
        self.tvolt.du_count=1
        self.tvolt.du_id=[1]
        self.tvolt.du_seconds=[1]
        self.tvolt.du_nanoseconds=[1]
        self.tvolt.trigger_flag=[1]
        self.tvolt.du_acceleration=[[1,2,3]]
        self.tvolt.trigger_rate=[1]
        self.tvolt.trace=[[[1,2,3]]]
        self.tvolt.p2p=[[1,2,3]]
        self.tvolt.time_max=[[1,2,3]]

    def test_tefield(self):
        self.tefield = groot.TEfield()
        self.assertTrue(self.tefield._file is None)
        self.assertTrue(hasattr(self.tefield, 'time_seconds'))
        self.assertTrue(hasattr(self.tefield, 'time_nanoseconds'))
        self.assertTrue(hasattr(self.tefield, 'event_type'))
        self.assertTrue(hasattr(self.tefield, 'du_count'))
        self.assertTrue(hasattr(self.tefield, 'du_id'))
        self.assertTrue(hasattr(self.tefield, 'du_seconds'))
        self.assertTrue(hasattr(self.tefield, 'du_nanoseconds'))
        self.assertTrue(hasattr(self.tefield, 'trace'))
        self.assertTrue(hasattr(self.tefield, 'fft_mag'))
        self.assertTrue(hasattr(self.tefield, 'fft_phase'))
        self.assertTrue(hasattr(self.tefield, 'p2p'))
        self.assertTrue(hasattr(self.tefield, 'pol'))
        self.assertTrue(hasattr(self.tefield, 'time_max'))
        self.tefield.time_seconds=1
        self.tefield.time_nanoseconds=1
        self.tefield.event_type=1
        self.tefield.du_count=1
        self.tefield.du_id=[1]
        self.tefield.du_seconds=[1]
        self.tefield.du_nanoseconds=[1]
        self.tefield.trace=[[[1,2,3]]]
        self.tefield.fft_mag=[[[1,2,3]]]
        self.tefield.fft_phase=[[[1,2,3]]]
        self.tefield.p2p=[[1,2,3]]
        self.tefield.pol=[[1,2,3]]
        self.tefield.time_max=[[1,2,3]]

    def test_tshower(self):
        self.tshower = groot.TShower()
        self.assertTrue(self.tshower._file is None)
        self.assertTrue(hasattr(self.tshower, 'primary_type'))
        self.assertTrue(hasattr(self.tshower, 'energy_em'))
        self.assertTrue(hasattr(self.tshower, 'energy_primary'))
        self.assertTrue(hasattr(self.tshower, 'azimuth'))
        self.assertTrue(hasattr(self.tshower, 'zenith'))
        self.assertTrue(hasattr(self.tshower, 'direction'))
        self.assertTrue(hasattr(self.tshower, 'shower_core_pos'))
        self.assertTrue(hasattr(self.tshower, 'atmos_model'))
        self.assertTrue(hasattr(self.tshower, 'atmos_model_param'))
        self.assertTrue(hasattr(self.tshower, 'magnetic_field'))
        self.assertTrue(hasattr(self.tshower, 'core_alt'))
        self.assertTrue(hasattr(self.tshower, 'xmax_grams'))
        self.assertTrue(hasattr(self.tshower, 'xmax_pos'))
        self.assertTrue(hasattr(self.tshower, 'xmax_pos_shc'))
        self.assertTrue(hasattr(self.tshower, 'core_time_s'))
        self.assertTrue(hasattr(self.tshower, 'core_time_ns'))
        self.tshower.primary_type='H'
        self.tshower.energy_em=1
        self.tshower.energy_primary=1
        self.tshower.azimuth=1
        self.tshower.zenith=1
        self.tshower.direction=[1,2,3]
        self.tshower.shower_core_pos=[1,2,3]
        self.tshower.atmos_model='x'
        self.tshower.atmos_model_param=[1,2,3]
        self.tshower.magnetic_field=[1,2,3]
        self.tshower.core_alt=1
        self.tshower.xmax_grams=1
        self.tshower.xmax_pos=[1,2,3]
        self.tshower.xmax_pos_shc=[1,2,3]
        self.tshower.core_time_s=1
        self.tshower.core_time_ns=1

    def test_trunefieldsim(self):
        self.trunesim = groot.TRunEfieldSim()
        self.assertTrue(self.trunesim._file is None)
        self.assertTrue(hasattr(self.trunesim, 'refractivity_model'))
        self.assertTrue(hasattr(self.trunesim, 'refractivity_model_parameters'))
        self.assertTrue(hasattr(self.trunesim, 't_pre'))
        self.assertTrue(hasattr(self.trunesim, 't_post'))
        self.assertTrue(hasattr(self.trunesim, 'sim_name'))
        self.assertTrue(hasattr(self.trunesim, 'sim_version'))
        self.trunesim.refractivity_model='x'
        self.trunesim.refractivity_model_parameters=[1]
        self.trunesim.t_pre=1
        self.trunesim.t_post=1
        self.trunesim.sim_name='x'
        self.trunesim.sim_version='x'



    def test_trunnoise(self):
        self.trnoise = groot.TRunNoise()
        self.assertTrue(self.trnoise._file is None)
        self.assertTrue(hasattr(self.trnoise, 'gal_noise_map'))
        self.assertTrue(hasattr(self.trnoise, 'gal_noise_LST'))
        self.assertTrue(hasattr(self.trnoise, 'gal_noise_sigma'))
        self.trnoise.gal_noise_map='x'
        self.trnoise.gal_noise_LST=1
        self.trnoise.gal_noise_sigma=[[1,2,3]]

    @unittest.skip("test_efield.root doesn't match expected naming convention (XXX_YYY_LZ_*.root). "
                   "DataDirectory.split_filenames expects at least 3 underscore-separated parts.")
    def test_datadirectory(self):
        ddir_path = Path(grand_get_path_root_pkg()) / "data"
        self.assertTrue((ddir_path).exists())
        ddir = groot.DataDirectory(dir_name=str(ddir_path))

    #def test_datafile(self):
    #    efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    #    self.assertTrue((efield_file).exists())
    #    fi = groot.DataFile(str(efield_file))
    #    self.assertTrue(len(fi.list_of_trees)>0)



if __name__ == "__main__":
    unittest.main()
