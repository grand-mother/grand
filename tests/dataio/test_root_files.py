"""
Unit tests for the grand.dataio.root_files module
Converted to pytest format for better test isolation and fixtures.
"""

import pytest
from pathlib import Path

import grand.dataio.root_files as RFile
import grand.dataio as groot
from grand import grand_get_path_root_pkg

# TODO: (JMC) almost all tests are broken by new version of GRANDROOT file, needs to have a set of coherent ROOT for test


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def efield_file():
    """Path to the test efield ROOT file."""
    return Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"


@pytest.fixture(scope="module")
def voltage_file():
    """Path to the test voltage ROOT file."""
    return Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root"


@pytest.fixture(scope="module")
def expected_shape():
    """Expected shape of traces: (du_count, 3, trace_size)."""
    return (96, 3, 999)


# ============================================================================
# Tests for _FileEventBase
# ============================================================================

class TestFileEventBase:

    """Tests for the _FileEventBase class."""

    @pytest.mark.skip(reason="test_efield.root doesn't match expected naming convention (XXX_YYY_LZ_*.root). "
                             "DataDirectory.split_filenames expects at least 3 underscore-separated parts.")
    def test_fileeventbase_initialization_and_loading(self, efield_file):
        """Test _FileEventBase initialization, event loading, and metadata."""
        assert efield_file.exists(), f"Test file {efield_file} does not exist"
        
        E = groot.TEfield(str(efield_file))
        eventbase = RFile._FileEventBase(E, str(efield_file))
        
        # Check attributes exist
        assert hasattr(eventbase, 'run_number')
        assert hasattr(eventbase, 'event_number')
        
        # Initially None before loading
        assert eventbase.run_number is None
        
        # Load first event
        eventbase.load_event_idx(0)
        assert eventbase.run_number == 0
        assert eventbase.event_number == 1
        
        # No more events
        assert eventbase.load_next_event() is False
        
        # Load by identifier
        eventbase._load_event_identifier(1, 0)
        assert eventbase.get_du_count() == 96
        assert eventbase.get_nb_events() == 1
        assert eventbase.get_size_trace() == 999
        assert eventbase.get_sampling_freq_mhz() == 2000


# ============================================================================
# Tests for get_file_event factory function
# ============================================================================

class TestGetFileEvent:

    """Tests for the get_file_event factory function."""

    @pytest.mark.skip(reason="test_efield.root doesn't contain expected TTree structure. "
                             "File needs proper tefield TTree to pass get_file_event factory check.")
    def test_get_file_event_with_efield(self, efield_file, expected_shape):
        """Test get_file_event with Efield file (always available)."""
        assert efield_file.exists(), f"Test file {efield_file} does not exist"
        
        E = RFile.get_file_event(str(efield_file))
        
        assert isinstance(E.run_number, int)
        assert isinstance(E.event_number, int)
        assert E.traces.shape == expected_shape
        assert E.traces.shape[-1] == expected_shape[-1]
        assert len(E.du_id) == expected_shape[0]
        assert E.du_count == expected_shape[0]

    @pytest.mark.skipif(
        not (Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root").exists(),
        reason="test_voltage.root not available. Generate with: "
               "python scripts/convert_efield2voltage.py data/test_efield.root -o data/test_voltage.root"
    )
    def test_get_file_event_with_voltage(self, efield_file, voltage_file, expected_shape):
        """Test get_file_event with both Efield and Voltage files."""
        assert efield_file.exists()
        assert voltage_file.exists()

        E = RFile.get_file_event(str(efield_file))
        V = RFile.get_file_event(str(voltage_file))
        
        assert isinstance(E.run_number, int) and isinstance(V.run_number, int)
        assert isinstance(E.event_number, int) and isinstance(V.event_number, int)
        assert E.traces.shape == expected_shape and V.traces.shape == expected_shape
        assert E.traces.shape[-1] == expected_shape[-1] and V.traces.shape[-1] == expected_shape[-1]
        assert len(E.du_id) == expected_shape[0] and len(V.du_id) == expected_shape[0]
        assert E.du_count == expected_shape[0] and V.du_count == expected_shape[0]

    @pytest.mark.skip(reason="test_efield.root doesn't contain expected TTree structure. "
                             "File needs proper tefield TTree to pass get_file_event factory check.")
    def test_get_file_event_returns_efield(self, efield_file):
        """Test get_file_event correctly identifies and returns FileEfield."""
        assert efield_file.exists()
        efield = RFile.get_file_event(str(efield_file))
        assert hasattr(efield, 'traces')


# ============================================================================
# Tests for FileEfield class
# ============================================================================

class TestFileEfield:

    """Tests for the FileEfield class."""

    @pytest.mark.skip(reason="test_efield.root doesn't match expected naming convention. "
                             "FileEfield internally creates DataDirectory which requires proper filename format.")
    def test_file_efield_initialization_and_attributes(self, efield_file, expected_shape):
        """Test FileEfield class initialization and attribute access."""
        assert efield_file.exists()

        E = RFile.FileEfield(str(efield_file))
        
        assert isinstance(E.run_number, int)
        assert isinstance(E.event_number, int)
        assert E.traces.shape == expected_shape
        assert E.traces.shape[-1] == expected_shape[-1]
        assert len(E.du_id) == expected_shape[0]
        assert E.du_count == expected_shape[0]


# ============================================================================
# Tests for FileVoltage class
# ============================================================================

class TestFileVoltage:


    """Tests for the FileVoltage class."""

    @pytest.mark.skipif(
        not (Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root").exists(),
        reason="test_voltage.root not available. Generate with: "
               "python scripts/convert_efield2voltage.py data/test_efield.root -o data/test_voltage.root"
    )
    def test_file_voltage_initialization_and_attributes(self, voltage_file, expected_shape):
        """Test FileVoltage class initialization and attribute access."""
        assert voltage_file.exists()

        V = RFile.FileVoltage(str(voltage_file))
        
        assert isinstance(V.run_number, int)
        assert isinstance(V.event_number, int)
        assert V.traces.shape == expected_shape
        assert V.traces.shape[-1] == expected_shape[-1]
        assert len(V.du_id) == expected_shape[0]
        assert V.du_count == expected_shape[0]


# ============================================================================
# Tests for helper functions
# ============================================================================

@pytest.mark.skip(reason="test_efield.root doesn't contain expected TTree structure. "
                         "Requires proper ROOT file with tefield TTree.")
def test_get_obj_handling3dtraces(efield_file):
    """Test get_obj_handling3dtraces function."""
    ef_root = RFile.get_file_event(str(efield_file))
    ef_obj = ef_root.get_obj_handling3dtraces()
    assert ef_obj.get_size_trace() == ef_root.sig_size
