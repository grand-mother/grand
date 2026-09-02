"""
Integration tests for grand.dataio package.

Tests complete end-to-end workflows using real ROOT file operations.
These tests verify that all components work together correctly.
"""

import pytest
import numpy as np
from pathlib import Path

from grand.dataio import TRun, TADC, TVoltage, TEfield, TShower
from grand.dataio import DataDirectory, DataFile


# ============================================================================
# Write-Read Cycle Tests
# ============================================================================

@pytest.mark.integration
class TestWriteReadCycle:
    """Tests for complete write-read cycles."""

    def test_trun_create_fill_write_read(self, tmp_path):
        """Test TRun complete workflow."""
        filepath = tmp_path / "trun_cycle.root"

        # Create and populate
        trun1 = TRun(_file_name=str(filepath))
        trun1.run_number = 42
        trun1.run_mode = 1
        trun1.data_source = "test"
        trun1.site = "test_site"

        # Fill and write
        trun1.fill()
        trun1.write()
        trun1.close_file()

        # Read back
        trun2 = TRun(_file_name=str(filepath))
        assert trun2.get_entries() == 1
        trun2.get_entry(0)
        assert trun2.run_number == 42

    def test_tadc_create_fill_write_read(self, tmp_path):
        """Test TADC complete workflow."""
        filepath = tmp_path / "tadc_cycle.root"

        # Create and populate
        tadc1 = TADC(_file_name=str(filepath))
        tadc1.run_number = 0
        tadc1.event_number = 1
        tadc1.du_count = 2

        # Fill and write
        tadc1.fill()
        tadc1.write()
        tadc1.close_file()

        # Read back
        tadc2 = TADC(_file_name=str(filepath))
        assert tadc2.get_entries() == 1
        tadc2.get_entry(0)
        assert tadc2.run_number == 0
        assert tadc2.event_number == 1

    def test_tvoltage_create_fill_write_read(self, tmp_path):
        """Test TVoltage complete workflow."""
        filepath = tmp_path / "tvoltage_cycle.root"

        # Create and populate
        tvolt1 = TVoltage(_file_name=str(filepath))
        tvolt1.run_number = 0
        tvolt1.event_number = 1
        tvolt1.du_count = 1

        # Fill and write
        tvolt1.fill()
        tvolt1.write()
        tvolt1.close_file()

        # Read back
        tvolt2 = TVoltage(_file_name=str(filepath))
        assert tvolt2.get_entries() == 1

    def test_multiple_events_workflow(self, tmp_path):
        """Test workflow with multiple events."""
        filepath = tmp_path / "multi_event.root"

        # Create and fill multiple events
        tadc = TADC(_file_name=str(filepath))
        for i in range(5):
            tadc.run_number = 0
            tadc.event_number = i
            tadc.du_count = 1
            tadc.fill()

        tadc.write()
        tadc.close_file()

        # Read and verify
        tadc2 = TADC(_file_name=str(filepath))
        assert tadc2.get_entries() == 5

        # Get specific event
        result = tadc2.get_event(ev_no=2, run_no=0)
        assert result > 0


# ============================================================================
# Multi-Tree Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestMultiTreeWorkflow:
    """Tests for workflows with multiple trees."""

    def test_trun_and_tadc_together(self, tmp_path):
        """Test using TRun and TADC together."""
        run_file = tmp_path / "run.root"
        adc_file = tmp_path / "adc.root"

        # Create TRun
        trun = TRun(_file_name=str(run_file))
        trun.run_number = 0
        trun.run_mode = 1
        trun.fill()
        trun.write()
        trun.close_file()

        # Create TADC for same run
        tadc = TADC(_file_name=str(adc_file))
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.du_count = 1
        tadc.fill()
        tadc.write()
        tadc.close_file()

        # Verify both exist
        assert run_file.exists()
        assert adc_file.exists()

    def test_metadata_persistence(self, tmp_path):
        """Test that metadata persists across write/read."""
        filepath = tmp_path / "metadata.root"

        # Create with metadata
        trun1 = TRun(_file_name=str(filepath))
        trun1.run_number = 0
        trun1.comment = "Integration test"
        trun1.analysis_level = 1
        trun1.modification_software = "pytest"
        trun1.fill()
        trun1.write()
        trun1.close_file()

        # Read and check metadata
        trun2 = TRun(_file_name=str(filepath))
        assert trun2.comment == "Integration test"
        assert trun2.analysis_level == 1
        assert trun2.modification_software == "pytest"


# ============================================================================
# Directory Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestDirectoryWorkflow:
    """Tests for directory-level workflows."""

    def test_create_files_scan_directory(self, tmp_path):
        """Test creating files then scanning with DataDirectory."""
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

    def test_multiple_runs_in_directory(self, tmp_path):
        """Test directory with multiple runs."""
        # Create files for different runs
        for run_no in range(3):
            filepath = tmp_path / f"trun_L0_r{run_no}.root"
            trun = TRun(_file_name=str(filepath))
            trun.run_number = run_no
            trun.fill()
            trun.write()
            trun.close_file()

        # Scan directory
        data_dir = DataDirectory(str(tmp_path))
        assert len(data_dir.file_list) == 3


# ============================================================================
# Data Persistence Tests
# ============================================================================

@pytest.mark.integration
class TestDataPersistence:
    """Tests for data persistence and retrieval."""

    def test_index_survives_write_read(self, tmp_path):
        """Test that index information survives write/read cycle."""
        filepath = tmp_path / "index_test.root"

        # Create with multiple runs
        trun = TRun(_file_name=str(filepath))
        for i in range(3):
            trun.run_number = i
            trun.run_mode = 1
            trun.fill()

        trun.write()
        trun.close_file()

        # Read back and check indexing works
        trun2 = TRun(_file_name=str(filepath))
        runs = trun2.get_list_of_runs()
        assert len(runs) == 3
        assert 0 in runs
        assert 1 in runs
        assert 2 in runs

    def test_event_retrieval_by_index(self, tmp_path):
        """Test retrieving events by (run, event) index."""
        filepath = tmp_path / "event_index.root"

        # Create events
        tadc = TADC(_file_name=str(filepath))
        events = [(0, 1), (0, 2), (1, 1)]
        for run_no, event_no in events:
            tadc.run_number = run_no
            tadc.event_number = event_no
            tadc.du_count = 1
            tadc.fill()

        tadc.write()
        tadc.close_file()

        # Read and verify all events can be retrieved
        tadc2 = TADC(_file_name=str(filepath))
        event_list = tadc2.get_list_of_events()
        assert len(event_list) == 3


# ============================================================================
# Complex Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestComplexWorkflows:
    """Tests for complex realistic workflows."""

    def test_realistic_run_event_workflow(self, tmp_path):
        """Test realistic workflow with run and multiple events."""
        run_file = tmp_path / "run_0.root"
        event_file = tmp_path / "events_0.root"

        # Create run
        trun = TRun(_file_name=str(run_file))
        trun.run_number = 0
        trun.run_mode = 1
        trun.data_source = "detector"
        trun.site = "GP13"
        trun.fill()
        trun.write()
        trun.close_file()

        # Create multiple events for that run
        tadc = TADC(_file_name=str(event_file))
        for i in range(10):
            tadc.run_number = 0
            tadc.event_number = i
            tadc.du_count = 3
            tadc.fill()

        tadc.write()
        tadc.close_file()

        # Verify both files
        assert run_file.exists()
        assert event_file.exists()

        # Verify contents
        trun2 = TRun(_file_name=str(run_file))
        assert trun2.get_entries() == 1

        tadc2 = TADC(_file_name=str(event_file))
        assert tadc2.get_entries() == 10
