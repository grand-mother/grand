"""
Shared pytest fixtures for grand.dataio tests.

Provides:
- ROOT object mocks (TFile, TTree, vector, string)
- Data generators (dummy traces, events, runs)
- File management fixtures (tmp_path based)
- Tree factories for testing
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, Mock
import tempfile


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )


# ============================================================================
# ROOT Mocking Fixtures
# ============================================================================

class FakeROOTVector:
    """Lightweight mock of ROOT.vector for unit testing."""

    def __init__(self, vec_type):
        self.vec_type = vec_type
        self._data = []

    def size(self):
        return len(self._data)

    def push_back(self, val):
        self._data.append(val)

    def clear(self):
        self._data = []

    def erase(self, index):
        del self._data[index]

    def insert(self, index, value):
        self._data.insert(index, value)

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        self._data[idx] = val

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def assign(self, value):
        """Mock ROOT vector assign method."""
        if isinstance(value, FakeROOTVector):
            self._data = value._data.copy()
        elif isinstance(value, (list, np.ndarray)):
            self._data = list(value)
        else:
            self._data = [value]


class FakeROOTString:
    """Lightweight mock of ROOT.string for unit testing."""

    def __init__(self, value=""):
        self._value = str(value)

    def assign(self, value):
        self._value = str(value)

    def __str__(self):
        return self._value

    def __repr__(self):
        return f"FakeROOTString('{self._value}')"


class FakeROOTTNamed:
    """Mock of ROOT.TNamed for metadata storage."""

    def __init__(self, name, title=""):
        self._name = name
        self._title = title

    def GetName(self):
        return self._name

    def GetTitle(self):
        return self._title

    def SetTitle(self, title):
        self._title = title


class FakeROOTTParameter:
    """Mock of ROOT.TParameter for numeric metadata."""

    def __init__(self, param_type):
        self._type = param_type
        self._name = ""
        self._value = 0

    def __call__(self, name, value):
        self._name = name
        self._value = value
        return self

    def GetVal(self):
        return self._value

    def SetVal(self, value):
        self._value = value

    def GetName(self):
        return self._name


class FakeROOTTList:
    """Mock of ROOT.TList for user info storage."""

    def __init__(self):
        self._items = {}

    def Add(self, obj):
        if hasattr(obj, 'GetName'):
            self._items[obj.GetName()] = obj

    def FindObject(self, name):
        return self._items.get(name, None)

    def Remove(self, obj):
        if hasattr(obj, 'GetName'):
            self._items.pop(obj.GetName(), None)

    def At(self, idx):
        return list(self._items.values())[idx]


class FakeROOTTTree:
    """Mock of ROOT.TTree for unit testing."""

    def __init__(self, name="", title=""):
        self._name = name
        self._title = title
        self._entries = []
        self._branches = {}
        self._user_info = FakeROOTTList()
        self._current_file = None
        self._directory = None

    def GetName(self):
        return self._name

    def SetName(self, name):
        self._name = name

    def SetTitle(self, title):
        self._title = title

    def GetEntries(self):
        return len(self._entries)

    def GetEntry(self, entry_num):
        if 0 <= entry_num < len(self._entries):
            return 1
        return 0

    def Fill(self):
        self._entries.append({})
        return len(self._entries)

    def GetUserInfo(self):
        return self._user_info

    def SetDirectory(self, directory):
        self._directory = directory

    def GetCurrentFile(self):
        return self._current_file if self._current_file else FakeROOTTFile("", "")

    def Branch(self, name, obj, leaflist=""):
        self._branches[name] = obj
        return Mock()

    def SetBranchAddress(self, name, obj):
        self._branches[name] = obj

    def BuildIndex(self, major, minor=""):
        return 0

    def GetEntryWithIndex(self, major, minor=0):
        return 0

    def Draw(self, *args, **kwargs):
        return 0

    def Scan(self, *args, **kwargs):
        pass


class FakeROOTTFile:
    """Mock of ROOT.TFile for unit testing."""

    def __init__(self, name, mode="READ"):
        self._name = name
        self._mode = mode
        self._trees = {}
        self._is_open = True

    def GetName(self):
        return self._name

    def Get(self, tree_name):
        return self._trees.get(tree_name, None)

    def Write(self, *args):
        pass

    def Close(self):
        self._is_open = False

    def ReOpen(self, mode):
        self._mode = mode
        return True

    def IsOpen(self):
        return self._is_open

    def GetListOfKeys(self):
        return Mock()


class FakeROOTTChain(FakeROOTTTree):
    """Mock of ROOT.TChain for unit testing."""

    def __init__(self, name, title=""):
        super().__init__(name, title)
        self._files = []

    def Add(self, filename):
        self._files.append(filename)


@pytest.fixture
def mock_root_vector():
    """Factory for creating fake ROOT vectors."""
    def _factory(vec_type):
        return FakeROOTVector(vec_type)
    return _factory


@pytest.fixture
def mock_root_module(monkeypatch):
    """Mock the entire ROOT module for unit tests."""
    import sys

    # Create a mock ROOT module
    mock_root = Mock()
    mock_root.vector = lambda vec_type: lambda value=[]: FakeROOTVector(vec_type)
    mock_root.string = FakeROOTString
    mock_root.TNamed = FakeROOTTNamed
    mock_root.TParameter = FakeROOTTParameter
    mock_root.TFile = FakeROOTTFile
    mock_root.TTree = FakeROOTTTree
    mock_root.TChain = FakeROOTTChain
    mock_root.nullptr = None
    mock_root.TObject = Mock()
    mock_root.TObject.kWriteDelete = 1

    # Mock gROOT
    mock_groot = Mock()
    mock_groot.GetVersionInt = Mock(return_value=63600)  # ROOT 6.36
    mock_groot.GetListOfFiles = Mock(return_value=Mock(FindObject=Mock(return_value=None)))
    mock_groot.LoadMacro = Mock()
    mock_root.gROOT = mock_groot

    # Mock gErrorIgnoreLevel
    mock_root.gErrorIgnoreLevel = 0
    mock_root.kFatal = 5000

    # Mock TTree constants
    mock_root.TTree = FakeROOTTTree
    mock_root.TTree.kMaxEntries = 1000000000

    monkeypatch.setitem(sys.modules, 'ROOT', mock_root)
    return mock_root


# ============================================================================
# Data Generation Fixtures
# ============================================================================

@pytest.fixture
def dummy_trace_data():
    """Generate random but valid trace data for testing."""
    np.random.seed(42)  # Reproducible
    return {
        'n_events': 3,
        'n_dus': 5,
        'trace_length': 100,
        'traces_adc': np.random.randint(-20, 21, (3, 5, 4, 100), dtype=np.int16),
        'traces_voltage': np.random.randn(3, 5, 3, 100).astype(np.float32),
    }


@pytest.fixture
def minimal_run_data():
    """Generate minimal valid run data."""
    return {
        'run_number': 0,
        'run_mode': 1,
        'data_source': 'test',
        'site': 'test_site',
        'n_dus': 3,
        'du_ids': [100, 101, 102],
        'du_geoid': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'du_xyz': [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
    }


@pytest.fixture
def minimal_event_data():
    """Generate minimal valid event data."""
    return {
        'run_number': 0,
        'event_number': 1,
        'du_count': 2,
        'du_ids': [100, 101],
        'time_seconds': 1234567890,
        'time_nanoseconds': 500000000,
    }


# ============================================================================
# File Management Fixtures
# ============================================================================

@pytest.fixture
def temp_root_file(tmp_path):
    """Create a temporary ROOT file path with auto-cleanup."""
    filepath = tmp_path / "test.root"
    yield filepath
    # tmp_path handles cleanup automatically


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for tests."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    yield test_dir
    # tmp_path handles cleanup automatically


# ============================================================================
# Tree Factory Fixtures
# ============================================================================

@pytest.fixture
def trun_factory(minimal_run_data):
    """Factory to create minimal TRun instances for testing."""
    def _factory(**kwargs):
        # Import here to avoid circular dependencies
        from grand.dataio import TRun

        data = minimal_run_data.copy()
        data.update(kwargs)

        trun = TRun()
        trun.run_number = data['run_number']
        trun.run_mode = data['run_mode']
        trun.data_source = data['data_source']
        trun.site = data['site']
        trun.du_id = data['du_ids']
        trun.du_geoid = data['du_geoid']
        trun.du_xyz = data['du_xyz']

        return trun
    return _factory


@pytest.fixture
def tadc_factory(minimal_event_data, dummy_trace_data):
    """Factory to create minimal TADC instances for testing."""
    def _factory(**kwargs):
        from grand.dataio import TADC

        data = minimal_event_data.copy()
        data.update(kwargs)

        tadc = TADC()
        tadc.run_number = data['run_number']
        tadc.event_number = data['event_number']
        tadc.du_count = data['du_count']
        tadc.du_id = data['du_ids']
        tadc.du_seconds = [data['time_seconds']] * data['du_count']
        tadc.du_nanoseconds = [data['time_nanoseconds']] * data['du_count']

        # Add minimal trace data
        tadc.trace_ch = dummy_trace_data['traces_adc'][0, :data['du_count'], :, :].tolist()

        return tadc
    return _factory


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def numpy_types():
    """Provide common numpy dtypes for parametrized testing."""
    return [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ]


@pytest.fixture
def cpp_types():
    """Provide C++ type strings for parametrized testing."""
    return [
        'char', 'short', 'int', 'long long',
        'unsigned char', 'unsigned short', 'unsigned int', 'unsigned long long',
        'float', 'double'
    ]
