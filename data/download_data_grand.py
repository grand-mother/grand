#! /usr/bin/env python3
import tarfile
import os
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

# Remove old data
print("==============================")
print("Updating data model. Removing old directories...")
for dir_name in ["detector", "noise", "topography"]:
    dir_path = grand_add_path_data(dir_name)
    if osp.exists(dir_path):
        shutil.rmtree(dir_path)

# Download new data
print("==============================")
print(f"Downloading new data model ({repo_version}), please wait...")
try:
    request.urlretrieve(LINK_MODEL, tar_file)
    print("Successfully downloaded.")
except HTTPError as e:
    print(f"Download failed: {LINK_MODEL}")
    print(f"HTTP Error: {e.code} - {e.reason}")
    sys.exit(1)
except URLError as e:
    print(f"Download failed: {LINK_MODEL}")
    print(f"Network Error: {e.reason}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during download: {e}")
    sys.exit(1)

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