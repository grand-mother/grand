#! /usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Downloads the GRAND detector data model, if it is not already current.

Compares the version in ``data/model_version.flag`` against the copy recorded
inside ``data/detector/`` and fetches the archive only when they differ.  Run
by ``env/setup.sh``; safe to run again at any time.

The archive is about 976 MB and is served from a single host,
``forge.in2p3.fr``, with no mirror, so transient failures are routine.  The
download is retried with exponential backoff, checked against the reported
content length -- a truncated transfer otherwise surfaces later as a confusing
tar error -- and staged to a temporary file so that a failure leaves the
existing installation intact.
"""

import tarfile
import os
import time
import sys
import shutil
import os.path as osp
from urllib import request
from urllib.error import URLError, HTTPError
from grand import GRAND_DATA_PATH, grand_add_path_data

# Paths for flag files
REPO_FLAG_FILE = grand_add_path_data('model_version.flag')  # Inside the repository
DETECTOR_DIR = grand_add_path_data('detector')
DETECTOR_FLAG_FILE = grand_add_path_data('detector/model_version.flag')  # Inside the detector directory

# Check if model_version.flag exists
if not osp.exists(REPO_FLAG_FILE):
    print("Error: Repository flag file is missing!")
    sys.exit(1)

# Read the stored download ID and version number from model_version.flag
with open(REPO_FLAG_FILE, 'r') as f:
    try:
        repo_download_id, repo_version = f.read().strip().split()
    except ValueError:
        print("Error: model_version.flag should contain both download ID and version (e.g., '404532 20250313').")
        sys.exit(1)

# Construct the correct download link
FORGE_BASE_URL = "https://forge.in2p3.fr/attachments/download"
LINK_MODEL = f"{FORGE_BASE_URL}/{repo_download_id}/grand_model_{repo_version}.tar.gz"
FILE_MODEL = f"grand_model_{repo_version}.tar.gz"
tar_file = osp.join(GRAND_DATA_PATH, FILE_MODEL)

# Check if detector directory exists
if not osp.exists(DETECTOR_DIR):
    print("==============================")
    print("Detector directory does not exist. Triggering data model download.")
    detector_version = None  # Force download
else:
    # Check if detector flag exists and compare versions
    if osp.exists(DETECTOR_FLAG_FILE):
        with open(DETECTOR_FLAG_FILE, 'r') as f:
            detector_version = f.read().strip()
    else:
        detector_version = None

# If detector_version is missing or different from repo_version, update the data
if detector_version == repo_version:
    print("==============================")
    print("Skip download: data model is up to date.")
    sys.exit(0)

# Download the new data model *before* removing the old one.
#
# The order used to be the other way round, and it made a transient network
# failure destructive: the detector, noise and topography directories were
# deleted first, so a download that then failed left the installation with no
# data at all rather than with the previous version.  That is what turns a
# forge.in2p3.fr hiccup into a broken environment.
#
# The archive is about 976 MB and comes from a single host with no mirror, so
# partial and refused transfers are both routine.  Hence the retries, and the
# size check: urlretrieve does not raise on a truncated download, it just
# returns a short file, which then fails later as a confusing tar error.

RETRIES = 4
BACKOFF_SECONDS = 5


def _download_once(url, destination):
    """Fetches `url` to `destination`, verifying the transfer is complete.

    Parameters
    ----------
    url : str
        What to fetch.
    destination : str
        Where to write it.

    Raises
    ------
    URLError
        If the server reports a length and fewer bytes arrived.  Raised as a
        network error because that is what it is, and so the caller's retry
        logic treats it like any other.
    """
    path, headers = request.urlretrieve(url, destination)
    expected = headers.get("Content-Length")
    if expected is not None:
        expected = int(expected)
        actual = osp.getsize(path)
        if actual != expected:
            os.remove(path)
            raise URLError(
                f"retrieval incomplete: got only {actual} out of {expected} bytes"
            )


print("==============================")
print(f"Downloading new data model ({repo_version}), please wait...")

tmp_file = tar_file + ".part"
for attempt in range(1, RETRIES + 1):
    try:
        _download_once(LINK_MODEL, tmp_file)
        print("Successfully downloaded.")
        break
    except (HTTPError, URLError) as e:
        reason = (f"HTTP Error: {e.code} - {e.reason}"
                  if isinstance(e, HTTPError) else f"Network Error: {e.reason}")
        print(f"Download attempt {attempt} of {RETRIES} failed: {reason}")
        if osp.exists(tmp_file):
            os.remove(tmp_file)
        if attempt == RETRIES:
            print(f"Download failed: {LINK_MODEL}")
            print("The data model is served from a single host without a "
                  "mirror; if it is unreachable, try again later.")
            sys.exit(1)
        delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
        print(f"Retrying in {delay} s...")
        time.sleep(delay)
    except Exception as e:
        if osp.exists(tmp_file):
            os.remove(tmp_file)
        print(f"Unexpected error during download: {e}")
        sys.exit(1)

# Only now is it safe to discard what is already installed.
print("==============================")
print("Updating data model. Removing old directories...")
for dir_name in ["detector", "noise", "topography"]:
    dir_path = grand_add_path_data(dir_name)
    if osp.exists(dir_path):
        shutil.rmtree(dir_path)
os.replace(tmp_file, tar_file)

# Extract new data
print("==============================")
print("Extracting tar file...")
try:
    with tarfile.open(tar_file) as my_tar:
        my_tar.extractall(grand_add_path_data(''))
    os.remove(tar_file)  # Delete tar file after extraction
except Exception as e:
    print(f"Extract failed: {tar_file}")
    print(f"Error: {e}")
    sys.exit(1)

# Write new version to detector flag file
with open(DETECTOR_FLAG_FILE, 'w') as f:
    f.write(repo_version)

print("Data model updated successfully!")
sys.exit(0)