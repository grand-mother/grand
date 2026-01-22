"""
Unit tests for the grand.dataio.protocol module.

Tests the URL/SSL utilities for downloading BLOBs from GitHub releases.
"""

import pytest
from grand.dataio.protocol import InvalidBLOB, get


def test_get_valid_blob():
    """Test downloading a valid BLOB from the store."""
    blob = get("check.txt")
    assert blob == b"This is just a check\n"


def test_get_invalid_blob():
    """Test that InvalidBLOB is raised for non-existent files."""
    with pytest.raises(InvalidBLOB):
        get("nonexistent_file_that_does_not_exist")


def test_get_with_custom_tag():
    """Test downloading with a specific tag parameter."""
    # Using the default tag should work
    blob = get("check.txt", tag="101")
    assert blob == b"This is just a check\n"


def test_get_invalid_tag():
    """Test that InvalidBLOB is raised for invalid tag."""
    with pytest.raises(InvalidBLOB):
        get("check.txt", tag="nonexistent_tag_999999")
