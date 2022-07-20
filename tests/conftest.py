"""configuration file for pytest
"""

import os
import os.path as osp
from urllib import request

from grand import GRAND_DATA_PATH


GRAND_ROOT = os.getenv("GRAND_ROOT")
DEPOT_GITHUB = "https://github.com/grand-mother/store/releases/download/101"


def pytest_configure(config):
    """Run master_data_test main function prior to run pytests."""
    if not osp.exists(GRAND_DATA_PATH):
        os.mkdir(GRAND_DATA_PATH)
    grand_download_huge_file()
    # add logger message
    #mlg.create_output_for_logger("debug", log_stdout=True)


def grand_download_huge_file(depot=DEPOT_GITHUB):
    name_file = "HorizonAntenna_EWarm_leff_loaded.npy"
    download_huge_file(
        f"{depot}/{name_file}",
        osp.join(GRAND_DATA_PATH, name_file),
    )


def download_huge_file(url_path, grand_path):
    if not osp.isfile(grand_path):
        request.urlretrieve(url_path, grand_path)
        print(f"Successfully downloaded the file {grand_path}")
