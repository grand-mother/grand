# grand/dataio Test Suite

## Overview

Comprehensive pytest-based unit tests for the grand/dataio package.

## Structure

```
tests/dataio/
├── conftest.py                  ✅ COMPLETE - Shared fixtures and ROOT mocks
├── test_protocol.py             ✅ COMPLETE - URL/SSL utilities (converted to pytest)
├── test_descriptors.py          ✅ COMPLETE - Core descriptor classes
├── test_data_tree.py            ⏳ TODO - DataTree base class
├── test_run_trees.py            ⏳ TODO - Run-level tree classes
├── test_event_trees.py          ⏳ TODO - Event-level tree classes
├── test_data_handling.py        ⏳ TODO - DataDirectory/DataFile
├── test_root_files.py           📝 EXISTS (needs pytest conversion)
├── test_integration.py          ⏳ TODO - End-to-end workflows
└── README_TESTS.md              📖 This file
```

## Completed Files

### 1. conftest.py ✅

**Purpose**: Foundation for all tests with shared fixtures and ROOT mocks

**Key Features**:
- Fake ROOT class implementations (FakeROOTVector, FakeROOTTFile, FakeROOTTTree, etc.)
- Mock ROOT module fixture for unit testing without ROOT
- Data generation fixtures (dummy_trace_data, minimal_run_data, minimal_event_data)
- File management fixtures (temp_root_file, temp_directory)
- Tree factory fixtures (trun_factory, tadc_factory)
- Type fixtures (numpy_types, cpp_types)

**Usage**:
```python
def test_something(mock_root_module, tmp_path, dummy_trace_data):
    # mock_root_module patches ROOT globally
    # tmp_path provides temporary directory
    # dummy_trace_data provides test data
    pass
```

### 2. test_protocol.py ✅

**Purpose**: Test URL/SSL utilities for downloading BLOBs

**Tests**:
- `test_get_valid_blob()` - Download existing BLOB
- `test_get_invalid_blob()` - Handle missing files
- `test_get_with_custom_tag()` - Custom tag parameter
- `test_get_invalid_tag()` - Invalid tag handling

**Status**: Converted from unittest to pytest

### 3. test_descriptors.py ✅

**Purpose**: Test core descriptor classes (building blocks for all trees)

**Test Classes**:
- `TestStdVectorList` - 15+ tests for vector operations
- `TestStdVectorListDesc` - Descriptor protocol tests
- `TestTTreeScalarDesc` - Scalar descriptor tests
- `TestTTreeArrayDesc` - Array descriptor tests
- `TestStdString` / `TestStdStringDesc` - String descriptor tests
- `TestHelperFunctions` - Type conversion utilities
- `TestNotUniqueEvent` - Exception handling
- `TestTypeConversions` - Type mapping dictionaries

**Key Test Coverage**:
- Multi-dimensional vectors (1D, 2D, 3D, 4D)
- Type conversions (numpy ↔ ROOT)
- Descriptor protocol (__get__, __set__, __set_name__)
- Edge cases (empty vectors, type mismatches)
- Parametrized tests for all supported types

## Remaining Files to Create

### 4. test_data_tree.py ⏳

**Recommended structure**:

```python
"""
Unit tests for the grand.dataio.data_tree module.

Tests the DataTree base class that all tree classes inherit from.
"""

import pytest
import datetime
import numpy as np
from grand.dataio import DataTree

class TestDataTreeCreation:
    """Tests for DataTree creation modes."""

    def test_create_without_file(self):
        """Test creating tree without associated file."""
        pass

    def test_create_with_filename(self, tmp_path):
        """Test creating tree with new file."""
        pass

    def test_open_existing_file(self, tmp_path):
        """Test opening tree from existing file."""
        pass

class TestDataTreeMetadata:
    """Tests for metadata management."""

    def test_creation_datetime_setter(self):
        """Test setting creation_datetime."""
        pass

    def test_modification_software_getter_setter(self):
        """Test modification_software property."""
        pass

    # ... etc for all metadata fields

class TestDataTreeFileOperations:
    """Tests for file I/O operations."""

    def test_write_to_new_file(self, tmp_path):
        """Test writing tree to new ROOT file."""
        pass

    def test_write_to_existing_file(self, tmp_path):
        """Test appending tree to existing file."""
        pass

class TestDataTreeIteration:
    """Tests for iteration protocol."""

    def test_iter_over_entries(self):
        """Test __iter__ method."""
        pass

# ... more test classes
```

**Key areas to test**:
- Tree creation (with/without file, with/without name)
- Metadata management (all properties)
- File operations (write, read, update modes)
- Branch creation and assignment
- Iteration protocol
- Attribute validation

### 5. test_run_trees.py ⏳

**Recommended structure**:

```python
"""
Unit tests for the grand.dataio.run_trees module.

Tests all run-level tree classes:
- MotherRunTree (base class)
- TRun, TRunVoltage, TRunRawVoltage
- TRunEfieldSim, TRunShowerSim, TRunNoise
"""

import pytest
from grand.dataio import (
    MotherRunTree, TRun, TRunVoltage,
    TRunRawVoltage, TRunEfieldSim,
    TRunShowerSim, TRunNoise
)

class TestMotherRunTree:
    """Tests for MotherRunTree base class."""

    def test_fill(self, tmp_path):
        """Test fill() method."""
        pass

    def test_build_index(self):
        """Test build_index on run_number."""
        pass

    def test_get_run(self):
        """Test get_run() method."""
        pass

    def test_has_run(self):
        """Test has_run() method."""
        pass

    def test_get_list_of_runs(self):
        """Test get_list_of_runs()."""
        pass

    def test_unique_run_enforcement(self):
        """Test that duplicate runs raise NotUniqueEvent."""
        pass

class TestTRun:
    """Tests for TRun class."""

    def test_all_fields_exist(self):
        """Test that all expected fields are present."""
        trun = TRun()
        assert hasattr(trun, 'run_number')
        assert hasattr(trun, 'run_mode')
        assert hasattr(trun, 'data_source')
        # ... check all fields

    def test_field_assignment(self, minimal_run_data):
        """Test assigning values to fields."""
        trun = TRun()
        trun.run_number = minimal_run_data['run_number']
        trun.du_id = minimal_run_data['du_ids']
        # ... assign all fields

# Similar test classes for TRunVoltage, TRunRawVoltage, etc.
```

**Key areas to test**:
- All 7 run tree classes
- Field presence and types
- Run uniqueness validation
- Index building and querying
- fill() method

### 6. test_event_trees.py ⏳

**Recommended structure**:

```python
"""
Unit tests for the grand.dataio.event_trees module.

Tests all event-level tree classes:
- MotherEventTree (base class)
- TADC, TRawVoltage, TVoltage
- TEfield, TShower, TShowerSim
"""

import pytest
from grand.dataio import (
    MotherEventTree, TADC, TRawVoltage,
    TVoltage, TEfield, TShower, TShowerSim
)

class TestMotherEventTree:
    """Tests for MotherEventTree base class."""

    def test_fill(self):
        """Test fill() method with run/event numbers."""
        pass

    def test_build_index(self):
        """Test build_index on (run_number, event_number)."""
        pass

    def test_get_event(self):
        """Test get_event(ev_no, run_no)."""
        pass

    def test_has_event(self):
        """Test has_event(ev_no, run_no)."""
        pass

    def test_get_list_of_events(self):
        """Test get_list_of_events()."""
        pass

    def test_get_traces_lengths(self):
        """Test get_traces_lengths()."""
        pass

    def test_get_list_of_dus(self):
        """Test get_list_of_dus()."""
        pass

    def test_duplicate_event_detection(self):
        """Test that duplicate (run,event) pairs raise NotUniqueEvent."""
        pass

class TestTADC:
    """Tests for TADC class."""

    def test_all_fields_exist(self):
        """Test that all TADC fields are present."""
        tadc = TADC()
        assert hasattr(tadc, 'run_number')
        assert hasattr(tadc, 'event_number')
        assert hasattr(tadc, 'du_count')
        assert hasattr(tadc, 'du_id')
        assert hasattr(tadc, 'trace_ch')
        # ... check all 120+ fields

# Similar test classes for TRawVoltage, TVoltage, TEfield, TShower, TShowerSim
```

**Key areas to test**:
- All 7 event tree classes
- Field presence and types (100+ fields per class)
- Event uniqueness validation
- Index building and querying
- Trace operations
- Friend tree relationships

### 7. test_data_handling.py ⏳

**Recommended structure**:

```python
"""
Unit tests for the grand.dataio.data_handling module.

Tests DataDirectory and DataFile classes for high-level file management.
"""

import pytest
from grand.dataio import DataDirectory, DataFile

class TestDataDirectory:
    """Tests for DataDirectory class."""

    def test_finds_root_files(self, tmp_path):
        """Test scanning directory for ROOT files."""
        # Create dummy .root files
        # Initialize DataDirectory
        # Verify files found
        pass

    def test_analysis_level_filtering(self, tmp_path):
        """Test filtering by analysis_level parameter."""
        pass

    def test_tree_attribute_access(self, tmp_path):
        """Test dynamic tree attribute access (dir.tadc, dir.trun)."""
        pass

class TestDataFile:
    """Tests for DataFile class."""

    def test_single_file_mode(self, tmp_path):
        """Test opening single ROOT file."""
        pass

    def test_chain_mode(self, tmp_path):
        """Test creating TChain from multiple files."""
        pass

    def test_context_manager(self, tmp_path):
        """Test 'with' statement support."""
        pass

    def test_tree_type_detection(self):
        """Test automatic tree type detection."""
        pass
```

**Key areas to test**:
- Directory scanning
- Analysis level selection
- TChain creation
- Context manager protocol
- Tree type detection

### 8. test_root_files.py (convert existing) ⏳

**Action needed**: Convert existing unittest tests to pytest and extend

**New tests to add**:
- Factory function tests
- Event loading tests
- Trace synchronization tests
- Simulation parameter extraction

### 9. test_integration.py ⏳

**Recommended structure**:

```python
"""
Integration tests for grand.dataio package.

Tests complete end-to-end workflows using real ROOT file operations.
"""

import pytest
from grand.dataio import TRun, TADC, DataDirectory

class TestWriteReadCycle:
    """Tests for complete write-read cycles."""

    def test_trun_write_read(self, tmp_path, minimal_run_data):
        """Test creating TRun, writing, reading back."""
        # Create TRun
        # Populate fields
        # Write to file
        # Read back
        # Verify data matches
        pass

    def test_tadc_write_read(self, tmp_path, dummy_trace_data):
        """Test creating TADC, writing, reading back."""
        pass

class TestMultiFileWorkflow:
    """Tests for workflows with multiple files."""

    def test_directory_scan_and_access(self, tmp_path):
        """Test scanning directory and accessing multiple trees."""
        pass

# ... more integration tests
```

**Key workflows to test**:
- Create → Fill → Write → Read → Verify
- Multi-file directory workflows
- Analysis level progressions (L0 → L1 → L2)

## Running Tests

### Run all tests:
```bash
pytest tests/dataio/ -v
```

### Run specific test file:
```bash
pytest tests/dataio/test_descriptors.py -v
```

### Run with coverage:
```bash
pytest tests/dataio/ --cov=grand.dataio --cov-report=html
```

### Run only unit tests (fast, mocked):
```bash
pytest tests/dataio/ -m unit -v
```

### Run only integration tests:
```bash
pytest tests/dataio/ -m integration -v
```

## Test Markers

Use these markers to categorize tests:

```python
@pytest.mark.unit        # Pure Python, all mocked
@pytest.mark.integration # Uses temporary ROOT files
@pytest.mark.slow        # Long-running tests
@pytest.mark.requires_root # Needs ROOT installation
```

## Fixtures Reference

### From conftest.py:

**ROOT Mocks**:
- `mock_root_vector` - Factory for FakeROOTVector
- `mock_root_module` - Patches entire ROOT module

**Data Generators**:
- `dummy_trace_data` - Random trace arrays
- `minimal_run_data` - Minimal run metadata
- `minimal_event_data` - Minimal event metadata

**File Management**:
- `temp_root_file(tmp_path)` - Temporary .root file
- `temp_directory(tmp_path)` - Temporary directory

**Tree Factories**:
- `trun_factory(**kwargs)` - Create TRun instances
- `tadc_factory(**kwargs)` - Create TADC instances

**Utilities**:
- `numpy_types` - List of numpy dtypes
- `cpp_types` - List of C++ type strings

## Example Usage

```python
def test_example(mock_root_module, tmp_path, dummy_trace_data, tadc_factory):
    """Example test using multiple fixtures."""
    # Create TADC using factory
    tadc = tadc_factory(run_number=5, event_number=10)

    # Add trace data
    tadc.trace_ch = dummy_trace_data['traces_adc'][0]

    # Write to temporary file
    filepath = tmp_path / "test.root"
    tadc.fill()
    tadc.write(str(filepath))

    # Read back and verify
    tadc2 = TADC(str(filepath))
    tadc2.get_entry(0)
    assert tadc2.run_number == 5
    assert tadc2.event_number == 10
```

## Implementation Guidelines

1. **Isolation**: Each test should be independent
2. **Clarity**: Clear test names and docstrings
3. **Mocking**: Use mocks for unit tests, real files for integration
4. **Cleanup**: Always use `tmp_path` for file operations
5. **Assertions**: Use pytest's assert statements
6. **Parametrization**: Use `@pytest.mark.parametrize` for multiple scenarios
7. **Documentation**: Every test has a docstring explaining what it tests

## Coverage Goals

Target >85% code coverage for each module:
- descriptors.py: >90%
- protocol.py: 100%
- data_tree.py: >85%
- event_trees.py: >85%
- run_trees.py: >85%
- data_handling.py: >85%
- root_files.py: >85%

## Next Steps

To complete the test suite:

1. ✅ Review completed files (conftest.py, test_protocol.py, test_descriptors.py)
2. ⏳ Create test_data_tree.py following the recommended structure
3. ⏳ Create test_run_trees.py
4. ⏳ Create test_event_trees.py
5. ⏳ Create test_data_handling.py
6. ⏳ Convert test_root_files.py to pytest
7. ⏳ Create test_integration.py
8. 🎯 Run full test suite and check coverage
9. 📝 Fix any failing tests
10. ✅ Achieve >85% coverage target
