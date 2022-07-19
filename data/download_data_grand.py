#! /usr/bin/env python3

'''
Created on 19 juil. 2022

@author: Jean-Marc Colley, CNRS/IN2P3/LPNHE

'''
import tarfile
import os
import sys
import os.path as osp
from urllib import request

from grand import GRAND_DATA_PATH

LINK_MODEL = "https://forge.in2p3.fr/attachments/download/133380/grand_model_2207.tar.gz"
FILE_MODEL = "grand_model_2207.tar.gz"

# 1- test if download is necessary
tar_file = osp.join(GRAND_DATA_PATH, FILE_MODEL)
if not os.path.exists(tar_file):
    # 2- download
    print("Download data model for GRAND")
    try:
        request.urlretrieve(LINK_MODEL, tar_file)
        print("Successfully downloaded")
    except:
        print(f"download failed {LINK_MODEL}")
        sys.exit(1)
# 3- extract
print('Extract tar file')
try:
    my_tar = tarfile.open(tar_file)
    my_tar.extractall(GRAND_DATA_PATH) 
    my_tar.close()
except:
    print(f"Extract failed '{tar_file}'")
    sys.exit(1)
sys.exit(0) 
