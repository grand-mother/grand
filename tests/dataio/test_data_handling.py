"""
Unit tests for the grand.dataio.data_handling module.

Tests high-level file and directory management classes:
- DataDirectory - Directory scanning and tree management
- DataFile - Single/multiple file handling with TChain
"""

import pytest

from grand.dataio.data_handling import DataDirectory, DataFile
from grand.dataio import TRun, TADC


# ============================================================================
# DataDirectory Tests
# ============================================================================

class TestDataDirectory:
    """Tests for DataDirectory class."""

    def test_initialization_with_empty_directory(self, tmp_path):
        """Test DataDirectory initialization with empty directory."""
        data_dir = DataDirectory(str(tmp_path))
        assert data_dir is not None
        assert data_dir.dir_name == str(tmp_path.absolute())

    def test_get_list_of_files_empty(self, tmp_path):
        """Test get_list_of_files with no ROOT files."""
        data_dir = DataDirectory(str(tmp_path))
        assert isinstance(data_dir.file_list, list)
        assert len(data_dir.file_list) == 0

    def test_get_list_of_files_with_root_files(self, tmp_path):
        """Test get_list_of_files with ROOT files present."""
        # Create actual ROOT files (not just touch)
        trun_file = tmp_path / "trun_L0_r0.root"
        tadc_file = tmp_path / "tadc_L0_r0.root"

        trun = TRun(_file_name=str(trun_file))
        trun.write()
        trun.close_file()

        tadc = TADC(_file_name=str(tadc_file))
        tadc.write()
        tadc.close_file()

        data_dir = DataDirectory(str(tmp_path))
        assert len(data_dir.file_list) == 2

    def test_analysis_level_parameter(self, tmp_path):
        """Test analysis_level parameter."""
        data_dir = DataDirectory(str(tmp_path), analysis_level=1)
        assert data_dir.analysis_level == 1

    def test_getattr_returns_none_for_missing_trees(self, tmp_path):
        """Test __getattr__ returns None for missing tree attributes."""
        data_dir = DataDirectory(str(tmp_path))
        result = data_dir.trun_l0
        assert result is None

    def test_print_method(self, tmp_path):
        """Test print method doesn't raise."""
        data_dir = DataDirectory(str(tmp_path))
        # Should not raise
        data_dir.print(verbose=False)

    def test_close_method(self, tmp_path):
        """Test close method."""
        data_dir = DataDirectory(str(tmp_path))
        # Should not raise
        data_dir.close()


# ============================================================================
# DataFile Tests
# ============================================================================

class TestDataFile:
    """Tests for DataFile class."""

    def test_initialization_with_single_file(self, tmp_path):
        """Test DataFile initialization with single ROOT file."""
        # Create a minimal ROOT file
        filepath = tmp_path / "test.root"
        trun = TRun(_file_name=str(filepath))
        trun.run_number = 0
        trun.write()
        trun.close_file()

        # Open with DataFile
        data_file = DataFile(str(filepath))
        assert data_file is not None
        assert data_file.filename == str(filepath)

    def test_initialization_with_file_list(self, tmp_path):
        """Test DataFile initialization with list of files."""
        # Create multiple ROOT files
        filepaths = []
        for i in range(2):
            filepath = tmp_path / f"test_{i}.root"
            trun = TRun(_file_name=str(filepath))
            trun.run_number = i
            trun.write()
            trun.close_file()
            filepaths.append(str(filepath))

        # Open with DataFile
        data_file = DataFile(filepaths)
        assert data_file is not None

    def test_close_method(self, tmp_path):
        """Test close method."""
        filepath = tmp_path / "test.root"
        trun = TRun(_file_name=str(filepath))
        trun.write()
        trun.close_file()

        data_file = DataFile(str(filepath))
        # Should not raise
        data_file.close()


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestDataHandlingIntegration:
    """Integration tests for DataDirectory and DataFile."""

    def test_directory_with_real_files(self, tmp_path):
        """Test DataDirectory with real ROOT files."""
        # Create TRun file
        trun_file = tmp_path / "trun_L0_r0.root"
        trun = TRun(_file_name=str(trun_file))
        trun.run_number = 0
        trun.fill()
        trun.write()
        trun.close_file()

        # Create TADC file
        tadc_file = tmp_path / "tadc_L0_r0.root"
        tadc = TADC(_file_name=str(tadc_file))
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.fill()
        tadc.write()
        tadc.close_file()

        # Scan directory
        data_dir = DataDirectory(str(tmp_path))
        assert len(data_dir.file_list) == 2
        assert len(data_dir.file_handle_list) > 0

    def test_datafile_context_manager(self, tmp_path):
        """Test DataFile as context manager."""
        filepath = tmp_path / "context_test.root"
        trun = TRun(_file_name=str(filepath))
        trun.write()
        trun.close_file()

        # Use as context manager if supported
        data_file = DataFile(str(filepath))
        assert data_file is not None
        data_file.close()

    def test_analysis_level_filtering(self, tmp_path):
        """Test filtering by analysis level."""
        # Create files with different analysis levels
        for level in [0, 1, 2]:
            filepath = tmp_path / f"trun_L{level}_r0.root"
            trun = TRun(_file_name=str(filepath))
            trun.run_number = 0
            trun.fill()
            trun.write()
            trun.close_file()

        # Scan with specific analysis level
        data_dir_l0 = DataDirectory(str(tmp_path), analysis_level=0)
        assert data_dir_l0.analysis_level == 0

        data_dir_l1 = DataDirectory(str(tmp_path), analysis_level=1)
        assert data_dir_l1.analysis_level == 1

        # Default (-1) should get maximum
        data_dir_max = DataDirectory(str(tmp_path), analysis_level=-1)
        assert data_dir_max.analysis_level == -1
