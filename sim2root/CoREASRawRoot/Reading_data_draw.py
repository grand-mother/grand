#!/usr/bin/env python
# coding: utf-8

# In[1]:


import grand.dataio.root_trees as rt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os.path
import glob
import numpy as np
from scipy.stats import norm
import ROOT
import datetime
import pandas as pd

from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
)
from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet

import numpy as np
import datetime
import time

import matplotlib.pyplot as plt
import os


# In[2]:


'''
GPS values of stations measured by a phone
'''
GPS_lat_true = {49: -35.1104783, 58: -35.112398, 59: -35.1123499, 60: -35.112391099999996, 69: -35.114382, 
               70: -35.114307 , 83: -35.1162916, 84: -35.1163491, 144: -35.110479999999995, 151: -35.1134566}
GPS_long_true = {49: -69.5308828, 58: -69.532292, 59: -69.52946829999999, 60: -69.52675889999999, 69: -69.530956, 
                70: -69.5281835, 83: -69.5294851, 84: -69.5267858, 144: -69.5274144, 151: -69.5253518 }


# In[3]:


file = "/home/grand/sim2root/CoREASRawRoot/Coreas_Run_4.root"
tadc  = rt.TADC(file)
df = rt.DataFile(file)

trawv = df.trawvoltage

count = trawv.draw('du_id',"")
du_id = np.unique(np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(int))
du_id = np.trim_zeros(du_id) #remove du 0 as it gives wrong data :(


GPS_lat = {}
GPS_long = {}
GPS_alt = {}
GPS_time = {}

mean_lats = []
mean_longs = []
mean_alts = []

for du in du_id:
    count = trawv.draw('gps_lat : gps_long : gps_alt : gps_time',"du_id == {}".format(du))
    
    gps_lat = np.array(np.frombuffer(trawv.get_v1(), count=count)).astype(float)
    gps_long = np.array(np.frombuffer(trawv.get_v2(), count=count)).astype(float)+360 # Grand cs does not accept negative longitudes
    gps_alt = np.array(np.frombuffer(trawv.get_v3(), count=count)).astype(float)
    gps_time = np.array(np.frombuffer(trawv.get_v4(), count=count)).astype(float)
    
    GPS_lat[du] = [gps_lat]
    GPS_long[du] = [gps_long]
    GPS_alt[du] = [gps_alt]
    GPS_time[du] = [gps_time]
    
    # To set the origin of the grand coordinate system we determine the mean of the locations
    mean_lats.append(np.mean(gps_lat))
    mean_longs.append(np.mean(gps_long))
    mean_alts.append(np.mean(gps_alt))


# In[4]:


mean_lat = np.mean(mean_lats)
mean_long = np.mean(mean_longs)
mean_alt = np.mean(mean_alts)
grand_origin = Geodetic(latitude=mean_lat, longitude=mean_long, height=mean_alt)


# In[5]:


for du in du_id:
    latitude = np.array(GPS_lat[du][0])
    longitude = np.array(GPS_long[du][0])
    altitude = np.array(GPS_alt[du][0]) 
        
    print("Du: {} \n Latitude: {} \n Longitude: {} \n \n".format(du, np.mean(latitude), np.mean(longitude)))


    # Conversion from Geodetic to GRAND coordinate system.
    geod = Geodetic(latitude=latitude, longitude=longitude, height=altitude)
    gcs = GRANDCS(geod, location=grand_origin)

    grand_x = (gcs.x) #/ 1000.0  # m -> km
    grand_y = (gcs.y) #/ 1000.0  # m -> km
        
        
    # Conversion of the 'TRUE' coordinates from Geodetic to GRAND coordinate system.
    lat_true = GPS_lat_true[du]
    long_true = GPS_long_true[du] 
        
    geod_true = Geodetic(latitude=lat_true, longitude=long_true+360, height=np.mean(altitude))
    gcs_true = GRANDCS(geod_true, location=grand_origin)

    grand_x_true = np.array((gcs_true.x)) 
    grand_y_true = np.array((gcs_true.y)) 
        
    # Conversion of the 'Origin coordinates from Geodetic to GRAND coordinate system.
    gcs_origin = GRANDCS(grand_origin, location=grand_origin)
    origin_x = np.array((gcs_origin.x))
    origin_y = np.array((gcs_origin.y))



    times = GPS_time[du][0]

    times_sort = np.argsort(times)

    times = np.array(times)[times_sort]
    grand_x = np.array(grand_x)[times_sort]
    grand_y = np.array(grand_y)[times_sort]
        

    time = []
    for t in times:
        time.append(datetime.datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S"))

    plt.figure(1)
    plt.plot(time, grand_x, label = 'du id = {}'.format(du))
    plt.title("X as function of time")
    plt.ylabel("X [m]")
    plt.xlabel("GPS time")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
        
    plt.figure(2)
    plt.plot(time, grand_y, label = 'du id = {}'.format(du))
    plt.title("Y as function of time")
    plt.ylabel("Y [m]")
    plt.xlabel("GPS time")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
        
    plt.figure(4)
    plt.hist(grand_x, histtype='step', label = 'du id = {}'.format(du))
    plt.title("X distribution")
    plt.xlabel("X(m)")
    plt.ylabel("counts")
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.figure(5)
    plt.hist(grand_y, histtype='step', label = 'du id = {}'.format(du))
    plt.title("Y distribution")
    plt.xlabel("Y(m)")
    plt.ylabel("counts")
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.figure(6)
    plt.scatter(grand_x, grand_y, label = 'du id = {}'.format(du))
    plt.scatter(origin_x, origin_y, color='black', marker='*')
    plt.gca().set_aspect("equal")
    plt.title("Grand @ Auger Antennae Positions du's in GRAND CS")
    plt.xlabel("North [m]")
    plt.ylabel("West [m]")
    plt.grid(ls="--", alpha=0.3)
    plt.legend()
        
    plt.figure(7)
    plt.scatter(grand_x_true, grand_y_true, label = 'du true = {}'.format(du))
    plt.scatter(origin_x, origin_y, color='black', marker='*')
    plt.gca().set_aspect("equal")
    plt.title("Grand @ Auger True Antennae Positions du's in GRAND CS")
    plt.xlabel("North [m]")
    plt.ylabel("West [m]")
    plt.grid(ls="--", alpha=0.3)
    plt.legend()
            
plt.show()

