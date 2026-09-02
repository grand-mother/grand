"""
Unit tests for the grand.dataio.run_trees module.

Tests all run-level tree classes:
- MotherRunTree (base class)
- TRun, TRunVoltage, TRunRawVoltage
- TRunEfieldSim, TRunShowerSim, TRunNoise
"""

import pytest

from grand.dataio.run_trees import (
    MotherRunTree,
    TRun,
    TRunVoltage,
    TRunRawVoltage,
    TRunEfieldSim,
    TRunShowerSim,
    TRunNoise
)
from grand.dataio.descriptors import NotUniqueEvent


# ============================================================================
# MotherRunTree Tests
# ============================================================================

class TestMotherRunTree:
    """Tests for MotherRunTree base class."""

    def test_initialization(self):
        """Test MotherRunTree can be initialized."""
        tree = MotherRunTree()
        assert tree is not None
        assert hasattr(tree, 'run_number')

    def test_fill_entry_list(self):
        """Test fill_entry_list method."""
        tree = MotherRunTree()
        tree.fill_entry_list()
        assert isinstance(tree._entry_list, list)

    def test_is_unique_event_returns_true_for_new_run(self):
        """Test is_unique_event returns True for new run."""
        tree = MotherRunTree()
        tree.run_number = 42
        assert tree.is_unique_event() == True

    def test_is_unique_event_returns_false_for_duplicate(self):
        """Test is_unique_event returns False for duplicate run."""
        tree = MotherRunTree()
        tree.run_number = 1
        tree._entry_list = [1, 2, 3]
        assert tree.is_unique_event() == False

    def test_get_list_of_runs_returns_list(self):
        """Test get_list_of_runs returns a list."""
        tree = MotherRunTree()
        runs = tree.get_list_of_runs()
        assert isinstance(runs, list)

    def test_has_run_method(self):
        """Test has_run method."""
        tree = MotherRunTree()
        # Should not raise
        result = tree.has_run(0)
        assert isinstance(result, bool)

    def test_build_index(self):
        """Test build_index method."""
        tree = MotherRunTree()
        # Should not raise
        tree.build_index("run_number")

    def test_print_list_of_runs(self):
        """Test print_list_of_runs method."""
        tree = MotherRunTree()
        # Should not raise
        tree.print_list_of_runs()


# ============================================================================
# TRun Tests
# ============================================================================

class TestTRun:
    """Tests for TRun class."""

    def test_initialization(self):
        """Test TRun can be initialized."""
        trun = TRun()
        assert trun is not None
        assert trun._tree_name == "trun"
        # Check that type property returns a string
        assert isinstance(trun.type, str)

    def test_run_number_field_exists(self):
        """Test run_number field exists."""
        trun = TRun()
        assert hasattr(trun, 'run_number')

    def test_run_mode_field_exists(self):
        """Test run_mode field exists."""
        trun = TRun()
        assert hasattr(trun, 'run_mode')

    def test_first_event_field_exists(self):
        """Test first_event field exists."""
        trun = TRun()
        assert hasattr(trun, 'first_event')

    def test_last_event_field_exists(self):
        """Test last_event field exists."""
        trun = TRun()
        assert hasattr(trun, 'last_event')

    def test_data_source_field_exists(self):
        """Test data_source field exists."""
        trun = TRun()
        assert hasattr(trun, 'data_source')

    def test_data_generator_field_exists(self):
        """Test data_generator field exists."""
        trun = TRun()
        assert hasattr(trun, 'data_generator')

    def test_event_type_field_exists(self):
        """Test event_type field exists."""
        trun = TRun()
        assert hasattr(trun, 'event_type')

    def test_site_field_exists(self):
        """Test site field exists."""
        trun = TRun()
        assert hasattr(trun, 'site')

    def test_origin_geoid_field_exists(self):
        """Test origin_geoid field exists."""
        trun = TRun()
        assert hasattr(trun, 'origin_geoid')

    def test_du_id_field_exists(self):
        """Test du_id field exists."""
        trun = TRun()
        assert hasattr(trun, 'du_id')

    def test_du_geoid_field_exists(self):
        """Test du_geoid field exists."""
        trun = TRun()
        assert hasattr(trun, 'du_geoid')

    def test_du_xyz_field_exists(self):
        """Test du_xyz field exists."""
        trun = TRun()
        assert hasattr(trun, 'du_xyz')

    def test_du_type_field_exists(self):
        """Test du_type field exists."""
        trun = TRun()
        assert hasattr(trun, 'du_type')

    def test_du_tilt_field_exists(self):
        """Test du_tilt field exists."""
        trun = TRun()
        assert hasattr(trun, 'du_tilt')

    def test_t_bin_size_field_exists(self):
        """Test t_bin_size field exists."""
        trun = TRun()
        assert hasattr(trun, 't_bin_size')

    def test_set_run_number(self):
        """Test setting run_number."""
        trun = TRun()
        trun.run_number = 42
        assert trun.run_number == 42

    def test_set_run_mode(self):
        """Test setting run_mode."""
        trun = TRun()
        trun.run_mode = 1
        assert trun.run_mode == 1

    def test_set_data_source(self):
        """Test setting data_source."""
        trun = TRun()
        trun.data_source = "detector"
        assert trun.data_source == "detector"

    def test_write_to_file(self, tmp_path):
        """Test writing TRun to file."""
        filepath = tmp_path / "trun_test.root"
        trun = TRun()
        trun.run_number = 0
        trun.run_mode = 1
        trun.write(str(filepath))
        assert filepath.exists()


# ============================================================================
# TRunVoltage Tests
# ============================================================================

class TestTRunVoltage:
    """Tests for TRunVoltage class."""

    def test_initialization(self):
        """Test TRunVoltage can be initialized."""
        trun_v = TRunVoltage()
        assert trun_v is not None

    def test_inherits_from_mother_run_tree(self):
        """Test TRunVoltage inherits from MotherRunTree."""
        trun_v = TRunVoltage()
        assert isinstance(trun_v, MotherRunTree)

    def test_has_run_number(self):
        """Test TRunVoltage has run_number field."""
        trun_v = TRunVoltage()
        assert hasattr(trun_v, 'run_number')


# ============================================================================
# TRunRawVoltage Tests
# ============================================================================

class TestTRunRawVoltage:
    """Tests for TRunRawVoltage class."""

    def test_initialization(self):
        """Test TRunRawVoltage can be initialized."""
        trun_rv = TRunRawVoltage()
        assert trun_rv is not None

    def test_inherits_from_mother_run_tree(self):
        """Test TRunRawVoltage inherits from MotherRunTree."""
        trun_rv = TRunRawVoltage()
        assert isinstance(trun_rv, MotherRunTree)

    def test_has_run_number(self):
        """Test TRunRawVoltage has run_number field."""
        trun_rv = TRunRawVoltage()
        assert hasattr(trun_rv, 'run_number')


# ============================================================================
# TRunEfieldSim Tests
# ============================================================================

class TestTRunEfieldSim:
    """Tests for TRunEfieldSim class."""

    def test_initialization(self):
        """Test TRunEfieldSim can be initialized."""
        trun_es = TRunEfieldSim()
        assert trun_es is not None

    def test_inherits_from_mother_run_tree(self):
        """Test TRunEfieldSim inherits from MotherRunTree."""
        trun_es = TRunEfieldSim()
        assert isinstance(trun_es, MotherRunTree)

    def test_has_simulation_fields(self):
        """Test TRunEfieldSim has simulation-specific fields."""
        trun_es = TRunEfieldSim()
        assert hasattr(trun_es, 'run_number')


# ============================================================================
# TRunShowerSim Tests
# ============================================================================

class TestTRunShowerSim:
    """Tests for TRunShowerSim class."""

    def test_initialization(self):
        """Test TRunShowerSim can be initialized."""
        trun_ss = TRunShowerSim()
        assert trun_ss is not None

    def test_inherits_from_mother_run_tree(self):
        """Test TRunShowerSim inherits from MotherRunTree."""
        trun_ss = TRunShowerSim()
        assert isinstance(trun_ss, MotherRunTree)

    def test_has_run_number(self):
        """Test TRunShowerSim has run_number field."""
        trun_ss = TRunShowerSim()
        assert hasattr(trun_ss, 'run_number')


# ============================================================================
# TRunNoise Tests
# ============================================================================

class TestTRunNoise:
    """Tests for TRunNoise class."""

    def test_initialization(self):
        """Test TRunNoise can be initialized."""
        trun_n = TRunNoise()
        assert trun_n is not None

    def test_inherits_from_mother_run_tree(self):
        """Test TRunNoise inherits from MotherRunTree."""
        trun_n = TRunNoise()
        assert isinstance(trun_n, MotherRunTree)

    def test_has_run_number(self):
        """Test TRunNoise has run_number field."""
        trun_n = TRunNoise()
        assert hasattr(trun_n, 'run_number')


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestRunTreesIntegration:
    """Integration tests for run tree classes."""

    def test_trun_fill_and_retrieve(self, tmp_path):
        """Test filling TRun and retrieving data."""
        filepath = tmp_path / "trun_integration.root"

        # Create and fill
        trun = TRun(_file_name=str(filepath))
        trun.run_number = 1
        trun.run_mode = 1
        trun.data_source = "test"
        trun.fill()

        # Write and close
        trun.write()
        trun.close_file()

        # Read back
        trun2 = TRun(_file_name=str(filepath))
        assert trun2 is not None
        assert trun2.get_entries() == 1

    def test_duplicate_run_raises_exception(self, tmp_path):
        """Test that duplicate runs raise NotUniqueEvent."""
        filepath = tmp_path / "duplicate_run.root"

        trun = TRun(_file_name=str(filepath))
        trun.run_number = 1
        trun.fill()

        # Try to fill same run again
        trun.run_number = 1
        with pytest.raises(NotUniqueEvent):
            trun.fill()

    def test_multiple_runs_in_tree(self, tmp_path):
        """Test storing multiple runs in single tree."""
        filepath = tmp_path / "multiple_runs.root"

        trun = TRun(_file_name=str(filepath))

        # Fill multiple runs
        for i in range(3):
            trun.run_number = i
            trun.run_mode = 1
            trun.fill()

        trun.write()
        trun.close_file()

        # Read back
        trun2 = TRun(_file_name=str(filepath))
        assert trun2.get_entries() == 3
