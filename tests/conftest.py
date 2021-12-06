"""configuration file for pytest
"""
import sys
import os
import os.path as osp

from urllib import request
import pytest
from grand import GRAND_DATA
from grand.simulation import TabulatedAntennaModel

GRAND_ROOT = os.getenv("GRAND_ROOT")


def pytest_configure(config):
    """Run master_data_test main function prior to run pytests.
    """
    if not osp.exists(GRAND_DATA): 
        os.mkdir(GRAND_DATA)
    grand_download_big_file()


def grand_download_big_file(): 
    name_file = "HorizonAntenna_EWarm_leff_loaded.npy"
    download_big_file(f"https://github.com/grand-mother/store/releases/download/101/{name_file}",
                      osp.join(GRAND_DATA, name_file))
    

def download_big_file(url_path, grand_path):
    if not osp.isfile(grand_path):
        request.urlretrieve(url_path, grand_path)
        print(f"Successfully downloaded the file {grand_path}")    
