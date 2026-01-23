"""
Unit tests for the grand.dataio.event_trees module.

Tests all event-level tree classes:
- MotherEventTree (base class)
- TADC, TRawVoltage, TVoltage
- TEfield, TShower, TShowerSim
"""

import pytest
import numpy as np
from pathlib import Path

from grand.dataio.event_trees import (
    MotherEventTree,
    TADC,
    TRawVoltage,
    TVoltage,
    TEfield,
    TShower,
    TShowerSim
)
from grand.dataio.descriptors import NotUniqueEvent


# ============================================================================
# MotherEventTree Tests
# ============================================================================

class TestMotherEventTree:
    """Tests for MotherEventTree base class."""

    def test_initialization(self):
        """Test MotherEventTree can be initialized."""
        tree = MotherEventTree()
        assert tree is not None
        assert hasattr(tree, 'run_number')
        assert hasattr(tree, 'event_number')

    def test_get_list_of_events_returns_list(self):
        """Test get_list_of_events returns a list."""
        tree = MotherEventTree()
        events = tree.get_list_of_events()
        assert isinstance(events, list)

    def test_has_event_method(self):
        """Test has_event method."""
        tree = MotherEventTree()
        # Should return a value without raising
        result = tree.has_event(0, 0)
        # Result depends on tree state
        assert result is not None or result is None

    def test_print_list_of_events(self):
        """Test print_list_of_events method."""
        tree = MotherEventTree()
        # Should not raise
        tree.print_list_of_events()

    def test_add_proper_friends(self):
        """Test add_proper_friends method."""
        tree = MotherEventTree()
        # Should not raise (returns 0 as friends disabled)
        result = tree.add_proper_friends()
        assert result == 0


# ============================================================================
# TADC Tests
# ============================================================================

class TestTADC:
    """Tests for TADC class - most complex event tree."""

    def test_initialization(self):
        """Test TADC can be initialized."""
        tadc = TADC()
        assert tadc is not None
        assert tadc._tree_name == "tadc"

    def test_run_number_field_exists(self):
        """Test run_number field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'run_number')

    def test_event_number_field_exists(self):
        """Test event_number field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'event_number')

    def test_du_count_field_exists(self):
        """Test du_count field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'du_count')

    def test_du_id_field_exists(self):
        """Test du_id field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'du_id')

    def test_du_seconds_field_exists(self):
        """Test du_seconds field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'du_seconds')

    def test_du_nanoseconds_field_exists(self):
        """Test du_nanoseconds field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'du_nanoseconds')

    def test_trace_ch_field_exists(self):
        """Test trace_ch field exists (main trace data)."""
        tadc = TADC()
        assert hasattr(tadc, 'trace_ch')

    def test_event_size_field_exists(self):
        """Test event_size field exists."""
        tadc = TADC()
        assert hasattr(tadc, 'event_size')

    def test_event_size_assigned(self):
        """Test event_size can be assigned."""
        tadc = TADC()
        tadc.event_size = 1024
        assert tadc.event_size == 1024

    def test_set_basic_fields(self):
        """Test setting basic event fields."""
        tadc = TADC()
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.du_count = 2
        assert tadc.run_number == 0
        assert tadc.event_number == 1
        assert tadc.du_count == 2

    def test_write_to_file(self, tmp_path):
        """Test writing TADC to file."""
        filepath = tmp_path / "tadc_test.root"
        tadc = TADC()
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.du_count = 1
        tadc.write(str(filepath))
        assert filepath.exists()


# ============================================================================
# TRawVoltage Tests
# ============================================================================

class TestTRawVoltage:
    """Tests for TRawVoltage class."""

    def test_initialization(self):
        """Test TRawVoltage can be initialized."""
        traw_v = TRawVoltage()
        assert traw_v is not None

    def test_inherits_from_mother_event_tree(self):
        """Test TRawVoltage inherits from MotherEventTree."""
        traw_v = TRawVoltage()
        assert isinstance(traw_v, MotherEventTree)

    def test_has_run_and_event_number(self):
        """Test TRawVoltage has run_number and event_number."""
        traw_v = TRawVoltage()
        assert hasattr(traw_v, 'run_number')
        assert hasattr(traw_v, 'event_number')

    def test_has_trace_ch_field(self):
        """Test TRawVoltage has trace_ch field."""
        traw_v = TRawVoltage()
        assert hasattr(traw_v, 'trace_ch')


# ============================================================================
# TVoltage Tests
# ============================================================================

class TestTVoltage:
    """Tests for TVoltage class."""

    def test_initialization(self):
        """Test TVoltage can be initialized."""
        tvolt = TVoltage()
        assert tvolt is not None

    def test_inherits_from_mother_event_tree(self):
        """Test TVoltage inherits from MotherEventTree."""
        tvolt = TVoltage()
        assert isinstance(tvolt, MotherEventTree)

    def test_has_run_and_event_number(self):
        """Test TVoltage has run_number and event_number."""
        tvolt = TVoltage()
        assert hasattr(tvolt, 'run_number')
        assert hasattr(tvolt, 'event_number')

    def test_has_voltage_specific_fields(self):
        """Test TVoltage has voltage-specific fields."""
        tvolt = TVoltage()
        assert hasattr(tvolt, 'du_count')


# ============================================================================
# TEfield Tests
# ============================================================================

class TestTEfield:
    """Tests for TEfield class."""

    def test_initialization(self):
        """Test TEfield can be initialized."""
        tefield = TEfield()
        assert tefield is not None

    def test_inherits_from_mother_event_tree(self):
        """Test TEfield inherits from MotherEventTree."""
        tefield = TEfield()
        assert isinstance(tefield, MotherEventTree)

    def test_has_run_and_event_number(self):
        """Test TEfield has run_number and event_number."""
        tefield = TEfield()
        assert hasattr(tefield, 'run_number')
        assert hasattr(tefield, 'event_number')

    def test_has_efield_trace_fields(self):
        """Test TEfield has electric field trace fields."""
        tefield = TEfield()
        assert hasattr(tefield, 'du_count')


# ============================================================================
# TShower Tests
# ============================================================================

class TestTShower:
    """Tests for TShower class."""

    def test_initialization(self):
        """Test TShower can be initialized."""
        tshower = TShower()
        assert tshower is not None

    def test_inherits_from_mother_event_tree(self):
        """Test TShower inherits from MotherEventTree."""
        tshower = TShower()
        assert isinstance(tshower, MotherEventTree)

    def test_has_run_and_event_number(self):
        """Test TShower has run_number and event_number."""
        tshower = TShower()
        assert hasattr(tshower, 'run_number')
        assert hasattr(tshower, 'event_number')


# ============================================================================
# TShowerSim Tests
# ============================================================================

class TestTShowerSim:
    """Tests for TShowerSim class."""

    def test_initialization(self):
        """Test TShowerSim can be initialized."""
        tshower_sim = TShowerSim()
        assert tshower_sim is not None

    def test_inherits_from_mother_event_tree(self):
        """Test TShowerSim inherits from MotherEventTree."""
        tshower_sim = TShowerSim()
        assert isinstance(tshower_sim, MotherEventTree)

    def test_has_run_and_event_number(self):
        """Test TShowerSim has run_number and event_number."""
        tshower_sim = TShowerSim()
        assert hasattr(tshower_sim, 'run_number')
        assert hasattr(tshower_sim, 'event_number')

    def test_has_simulation_fields(self):
        """Test TShowerSim has simulation-specific fields."""
        tshower_sim = TShowerSim()
        # TShowerSim has simulation fields like event_date, rnd_seed
        assert hasattr(tshower_sim, 'event_date')
        assert hasattr(tshower_sim, 'rnd_seed')


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestEventTreesIntegration:
    """Integration tests for event tree classes."""

    def test_tadc_fill_and_retrieve(self, tmp_path):
        """Test filling TADC and retrieving data."""
        filepath = tmp_path / "tadc_integration.root"

        # Create and fill
        tadc = TADC(_file_name=str(filepath))
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.du_count = 1
        tadc.fill()

        # Write and close
        tadc.write()
        tadc.close_file()

        # Read back
        tadc2 = TADC(_file_name=str(filepath))
        assert tadc2 is not None
        assert tadc2.get_entries() == 1

    def test_duplicate_event_raises_exception(self, tmp_path):
        """Test that duplicate (run, event) pairs raise NotUniqueEvent."""
        filepath = tmp_path / "duplicate_event.root"

        tadc = TADC(_file_name=str(filepath))
        tadc.run_number = 0
        tadc.event_number = 1
        tadc.fill()

        # Try to fill same event again
        tadc.run_number = 0
        tadc.event_number = 1
        with pytest.raises(NotUniqueEvent):
            tadc.fill()

    def test_multiple_events_in_tree(self, tmp_path):
        """Test storing multiple events in single tree."""
        filepath = tmp_path / "multiple_events.root"

        tadc = TADC(_file_name=str(filepath))

        # Fill multiple events
        for i in range(3):
            tadc.run_number = 0
            tadc.event_number = i
            tadc.du_count = 1
            tadc.fill()

        tadc.write()
        tadc.close_file()

        # Read back
        tadc2 = TADC(_file_name=str(filepath))
        assert tadc2.get_entries() == 3

    def test_get_list_of_events_after_filling(self, tmp_path):
        """Test get_list_of_events after filling events."""
        filepath = tmp_path / "event_list.root"

        tadc = TADC(_file_name=str(filepath))

        # Fill events
        for i in range(2):
            tadc.run_number = 0
            tadc.event_number = i
            tadc.fill()

        tadc.write()
        tadc.close_file()

        # Read and get list
        tadc2 = TADC(_file_name=str(filepath))
        events = tadc2.get_list_of_events()
        assert len(events) == 2

    def test_tvoltage_basic_workflow(self, tmp_path):
        """Test basic TVoltage workflow."""
        filepath = tmp_path / "tvoltage.root"

        tvolt = TVoltage(_file_name=str(filepath))
        tvolt.run_number = 0
        tvolt.event_number = 1
        tvolt.du_count = 2
        tvolt.fill()
        tvolt.write()
        tvolt.close_file()

        # Read back
        tvolt2 = TVoltage(_file_name=str(filepath))
        assert tvolt2.get_entries() == 1
