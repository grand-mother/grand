# Unittest to Pytest Conversion Reference

## Quick Reference: What Changed

### Import Changes

```python
# BEFORE (unittest)
import unittest
from tests import TestCase

# AFTER (pytest)
import pytest
```

### Class Definition

```python
# BEFORE
class RootFilesTest(TestCase):
    efield_file = Path(...)
    voltage_file = Path(...)
    
# AFTER
@pytest.fixture(scope="module")
def efield_file():
    return Path(...)

@pytest.fixture(scope="module")
def voltage_file():
    return Path(...)

class TestFileEventBase:
```

### Test Method Signatures

```python
# BEFORE
def test_something(self):
    ...

# AFTER
def test_something(self, efield_file, expected_shape):
    # Fixtures automatically injected
    ...
```

### Assertions

```python
# BEFORE (unittest)                    # AFTER (pytest)
self.assertTrue(x)                     assert x
self.assertFalse(x)                    assert not x
self.assertEqual(a, b)                 assert a == b
self.assertNotEqual(a, b)              assert a != b
self.assertIs(a, b)                    assert a is b
self.assertIsNone(x)                   assert x is None
self.assertIn(a, b)                    assert a in b
self.assertIsInstance(a, type)         assert isinstance(a, type)
self.assertRaises(Exception):          with pytest.raises(Exception):
```

### Skip Decorators

```python
# BEFORE
@unittest.skipUnless(condition, "reason")
def test_something(self):
    ...

# AFTER
@pytest.mark.skipif(not condition, reason="reason")
def test_something(self, fixture):
    ...
```

### Running Tests

```bash
# BEFORE (unittest)
python -m unittest tests.dataio.test_root_files -v
python -m unittest discover

# AFTER (pytest)
pytest tests/dataio/test_root_files.py -v
pytest tests/dataio/ -v
pytest -v  # all tests

# With coverage
coverage run --source=grand -m pytest tests/dataio/ -v
coverage report -m
```

### Test Organization

```python
# BEFORE - Flat structure
class RootFilesTest(TestCase):
    def test_fileeventbase(self): ...
    def test_get_file_event1(self): ...
    def test_FileEfield(self): ...
    def test_FileVoltage(self): ...

# AFTER - Organized by component
class TestFileEventBase:
    def test_initialization(self): ...
    def test_loading(self): ...

class TestGetFileEvent:
    def test_with_efield(self): ...
    def test_with_voltage(self): ...

class TestFileEfield:
    def test_initialization(self): ...

class TestFileVoltage:
    def test_initialization(self): ...
```

## Conversion Checklist

- [ ] Remove `import unittest` and `from tests import TestCase`
- [ ] Add `import pytest`
- [ ] Convert class attributes to `@pytest.fixture`
- [ ] Change class inheritance from `TestCase` to no inheritance
- [ ] Replace `self.assert*` with `assert` statements
- [ ] Convert `@unittest.skipUnless` to `@pytest.mark.skipif`
- [ ] Add fixture parameters to test methods
- [ ] Remove `if __name__ == "__main__": unittest.main()`
- [ ] Add docstrings to tests and fixtures
- [ ] Organize tests into logical classes

## Benefits Gained

✅ **Cleaner syntax** - Pure Python assertions  
✅ **Better error messages** - Pytest shows values in failed assertions  
✅ **Fixtures** - Better dependency management  
✅ **Parameterization** - Easy with `@pytest.mark.parametrize`  
✅ **Plugins** - Rich ecosystem (pytest-cov, pytest-xdist, etc.)  
✅ **Markers** - Categorize tests (`@pytest.mark.slow`, etc.)  
✅ **No self** - Cleaner function signatures  
✅ **Modern** - Industry standard for Python testing  

## Example: Complete Before/After

### Before (unittest)
```python
import unittest
from tests import TestCase
from pathlib import Path

class RootFilesTest(TestCase):
    efield_file = Path("/path/to/test.root")
    shape = (96, 3, 999)
    
    def test_file_efield(self):
        self.assertTrue(self.efield_file.exists())
        E = RFile.FileEfield(str(self.efield_file))
        self.assertEqual(E.traces.shape, self.shape)
        self.assertIsInstance(E.run_number, int)

if __name__ == "__main__":
    unittest.main()
```

### After (pytest)
```python
import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def efield_file():
    """Path to test efield file."""
    return Path("/path/to/test.root")

@pytest.fixture(scope="module")
def expected_shape():
    """Expected shape of traces."""
    return (96, 3, 999)

class TestFileEfield:
    """Tests for FileEfield class."""
    
    def test_file_efield_initialization(self, efield_file, expected_shape):
        """Test FileEfield initialization and attributes."""
        assert efield_file.exists()
        E = RFile.FileEfield(str(efield_file))
        assert E.traces.shape == expected_shape
        assert isinstance(E.run_number, int)
```

## Running with Coverage in Conda Environment

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate grandenv_2509

# Run tests with coverage
coverage run --source=grand -m pytest tests/dataio/ -v

# View coverage report
coverage report -m

# Generate HTML report
coverage html
```

## Common Pitfalls

❌ **Don't** keep `self.assert*` - Use `assert` instead  
❌ **Don't** inherit from `TestCase` - Pure Python classes  
❌ **Don't** use class attributes for test data - Use fixtures  
❌ **Don't** forget to add fixture parameters to test methods  
❌ **Don't** use `unittest.skipUnless` - Use `@pytest.mark.skipif`  

✅ **Do** use fixtures for shared test data  
✅ **Do** use descriptive test names  
✅ **Do** organize tests into logical classes  
✅ **Do** add docstrings  
✅ **Do** use `assert` statements  
