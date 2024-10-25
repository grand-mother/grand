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
#TODO: add progressbar to grand lib
#import progressbar

from grand import GRAND_DATA_PATH, grand_add_path_data

#LINK_MODEL = "https://forge.in2p3.fr/attachments/download/133380/grand_model_2207.tar.gz"
#FILE_MODEL = "grand_model_2207.tar.gz"
#LINK_MODEL = "https://forge.in2p3.fr/attachments/download/201909/grand_model_2306.tar.gz"
#LINK_MODEL = "https://forge.in2p3.fr/attachments/download/251637/grand_model_190224.tar.gz"
LINK_MODEL = "https://forge.in2p3.fr/attachments/download/326497/LFmap.tar.gz"
FILE_MODEL = LINK_MODEL.split("/")[-1]
# class MyProgressBar():
#     def __init__(self):
#         self.pbar = None
#
#     def __call__(self, block_num, block_size, total_size):
#         if not self.pbar:
#             self.pbar=progressbar.ProgressBar(maxval=total_size)
#             self.pbar.start()
#
#         downloaded = block_num * block_size
#         if downloaded < total_size:
#             self.pbar.update(downloaded)
#         else:
#             self.pbar.finish()


# 1- test if download is necessary
if os.path.exists(grand_add_path_data('noise/LFmap')):
    print("==============================")
    print('Skip download LFmap files for Galactic noise simulations')
    sys.exit(0)

GRAND_DATA_PATH_1 = osp.join(grand_add_path_data('noise'))
tar_file = osp.join(GRAND_DATA_PATH_1, FILE_MODEL)

# 2- download
print("==============================")
print("Download LFmap model (~ 6.4 MB) for GRAND, please wait ...")
try:
    request.urlretrieve(LINK_MODEL, tar_file)
    print("Successfully downloaded")
except:
    print(f"download failed {LINK_MODEL}")
    sys.exit(1)
    
# 3- extract
print("==============================")
print('Extract tar file')
try:
    my_tar = tarfile.open(tar_file)
    my_tar.extractall(grand_add_path_data('noise'))
    my_tar.close()
    os.remove(tar_file)  # delete zipped file are extraction.
except:
    print(f"Extract failed '{tar_file}'")
    sys.exit(1)
print("LFmap model available in grand/data/noise directory !")
sys.exit(0) 
