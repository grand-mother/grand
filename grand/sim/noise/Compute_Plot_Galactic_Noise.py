#!/usr/bin/python

# This script simulates galactic noise in both sites gp13, gaa. 
# Please run first the script /grand/data/python3 download_LFmap_grand.py in order to download the LFmap files which are used for the Galactic noise calculations. The LFmap files are stored in the directory /grand/data/noise/LFmap
# Computes Open circuit Voltage (before RF chain), output voltage (after RF chain), Power, 
# average temperature, radiance and Electric field. 
# Author: Stavros Nonis
#usage: Compute_plot_Galactic_Noise.py [-h] [--site {gp13,gaa}]
#                                      [--du_type {GP300,GP300_nec,GP300_mat}] --run_mode
#                                      {Voc,Vout,PL,Efield}
#                                      [--lst_value {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}]

#Process Galactic Noise Data
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --site {gp13,gaa}     Site location
#  --du_type {GP300,GP300_nec,GP300_mat}
#                        Detector unit type
#  --run_mode {Voc,Vout,PL,Efield}
#                        Run mode
#  --lst_value {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}
                #LST value for frequency plot
# Example of how to use:
#python3 Compute_plot_Galactic_Noise.py --site gp13 --du_type GP300 --run_mode Voc --lst_value 18
# For the Voc run_mode the output file that saved is in shape (221,72,3) wich means the Volts^2/Hz (square Vrms) for 221 #frequencies bins (30-250 MHz step 1MHz), 72 LST bins (0:00-23:40, step 20 min) for X, Y, Z ports.

# For the Vout run_mode the output file that saved is in shape (221,72,3) wich means the Volts^2/Hz (square Vrms) for #221 frequencies bins (30-250 MHz step 1MHz), 72 LST bins (0:00-23:40, step 20 min) for X, Y, Z ports.

# For the PL run_mode the output file that saved is in shape (221,72,3) wich means the Watt/Hz for 221 #frequencies bins (30-250 MHz step 1MHz), 72 LST bins (0:00-23:40, step 20 min) for X, Y, Z ports.

# For the Efield run_mode the output file that saved is in shape (221,72,3) wich means the <temp * kB> (J), the average radiance <B_nu> (W/m2/Hz) and the average Efiled_rms^2 (V^2/m^2/Hz) for 221 #frequencies bins (30-250 MHz step 1MHz), 72 LST bins (0:00-23:40, step 20 min).


import numpy as np
import argparse
import matplotlib.pyplot as plt
import grand.sim.detector.rf_chain as grfc
from grand import grand_add_path_data

def compute_rfchain_and_latitude(site):
    if site == "gp13":
        rfchain = grfc.RFChain(vga_gain=20)
        latitude = (90 - 40.98) * np.pi / 180
    elif site == "gaa":
        rfchain = grfc.RFChain_gaa()
        latitude = (90 + 36) * np.pi / 180
    else:
        raise ValueError("Unsupported site value!")
    return rfchain, latitude

def compute_antenna_paths(du_type):
    if du_type == "GP300":
        path_antX = grand_add_path_data("detector/Light_GP300Antenna_SNarm_leff.npz")
        path_antY = grand_add_path_data("detector/Light_GP300Antenna_EWarm_leff.npz")
        path_antZ = grand_add_path_data("detector/Light_GP300Antenna_Zarm_leff.npz")
    elif du_type == "GP300_nec":
        path_antX = grand_add_path_data("detector/Light_GP300Antenna_nec_Xarm_leff.npz")
        path_antY = grand_add_path_data("detector/Light_GP300Antenna_nec_Yarm_leff.npz")
        path_antZ = grand_add_path_data("detector/Light_GP300Antenna_nec_Zarm_leff.npz")
    elif du_type == "GP300_mat":
        path_antX = grand_add_path_data("detector/Light_GP300Antenna_mat_Xarm_leff.npz")
        path_antY = grand_add_path_data("detector/Light_GP300Antenna_mat_Yarm_leff.npz")
        path_antZ = grand_add_path_data("detector/Light_GP300Antenna_mat_Zarm_leff.npz")
    else:
        raise ValueError("Unsupported du_type value!")
    return path_antX, path_antY, path_antZ

def main(site, du_type, run_mode, lst_value):
    freq_MHz = np.arange(30, 251, 1)
    
    # Initialize RFChain and latitude based on site
    rfchain, latitude = compute_rfchain_and_latitude(site)
    rfchain.compute_for_freqs(freq_MHz)
    RFchainNS = np.array(rfchain.get_tf()[0])
    RFchainEW = np.array(rfchain.get_tf()[1])
    RFchainZ = np.array(rfchain.get_tf()[2])

    # Initialize antenna paths based on du_type
    path_antX, path_antY, path_antZ = compute_antenna_paths(du_type)
    f_leffX = np.load(path_antX)
    f_leffY = np.load(path_antY)
    f_leffZ = np.load(path_antZ)

    freqsleff = f_leffX["freq_mhz"]
    leffthX = np.moveaxis(np.abs(f_leffX["leff_theta"]), -1, 0)
    leffphX = np.moveaxis(np.abs(f_leffX["leff_phi"]), -1, 0)
    leffthY = np.moveaxis(np.abs(f_leffY["leff_theta"]), -1, 0)
    leffphY = np.moveaxis(np.abs(f_leffY["leff_phi"]), -1, 0)
    leffthZ = np.moveaxis(np.abs(f_leffZ["leff_theta"]), -1, 0)
    leffphZ = np.moveaxis(np.abs(f_leffZ["leff_phi"]), -1, 0)

    lefftX = np.zeros((len(freqsleff), 361, 181))
    leffpX = np.zeros((len(freqsleff), 361, 181))
    lefftY = np.zeros((len(freqsleff), 361, 181))
    leffpY = np.zeros((len(freqsleff), 361, 181))
    lefftZ = np.zeros((len(freqsleff), 361, 181))
    leffpZ = np.zeros((len(freqsleff), 361, 181))

    lefftX[:, :, :91] = leffthX
    leffpX[:, :, :91] = leffphX
    lefftY[:, :, :91] = leffthY
    leffpY[:, :, :91] = leffphY
    lefftZ[:, :, :91] = leffthZ
    leffpZ[:, :, :91] = leffphZ

    # Constants and frequency setup
    c = 299792458
    kB = 1.38064852e-23
    Z0 = 120 * np.pi
    dnu = 1  # MHz
    freqs = np.arange(30, 251, dnu)
    longitude = np.arange(180, 360 + 180, 5)
    lstaxis = (longitude - 180) / 15
    lon=longitude*np.pi/180 #rad
    
    # Initialize variables to store results
    voc2X = np.zeros((len(freqs), len(lstaxis)))
    voc2Y = np.zeros((len(freqs), len(lstaxis)))
    voc2Z = np.zeros((len(freqs), len(lstaxis)))
    vout2X = np.zeros_like(voc2X)
    vout2Y = np.zeros_like(voc2Y)
    vout2Z = np.zeros_like(voc2Z)
    
    avleff2X=np.zeros(len(freqs))
    avleff2Y=np.zeros(len(freqs))
    avleff2Z=np.zeros(len(freqs))
    avtemp=np.zeros(len(freqs))
    avBnu=np.zeros(len(freqs))
    avErms2=np.zeros(len(freqs))
    
    dphi=5 #deg
    dtheta=5 #deg
    zenith,azimuth=np.meshgrid( np.arange(0,180,dtheta)*np.pi/180, np.arange(0,360,dphi)*np.pi/180 )#rad
    nazimuth=72
    nzenith=36
    idselfreq=np.arange(0,len(freqsleff),dnu)
    idseltheta=np.arange(0,181,dphi)
    idselphi=np.arange(0,361,dtheta)
    lefftX=lefftX[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    leffpX=leffpX[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    lefftY=lefftY[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    leffpY=leffpY[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    lefftZ=lefftZ[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    leffpZ=leffpZ[idselfreq,:,:][:,idselphi,:][:,:,idseltheta]
    T=np.zeros((len(freqs),len(lon),72,36))
    print("Compute Galactic noise contribution in the frequency range 30-250 MHz")
    
    for f in range(len(freqs)):
        ra,dec,temp=np.load(grand_add_path_data("noise/LFmap/LFmapshort"+str(freqs[f])+".npy")) #rad inside
        integ=0
        print("f = ",freqs[f],"MHz")

        for l in range(len(lon)):
            omega=0
            for i in range(nazimuth):
                for j in range(nzenith):
                    #(RzRy)-1
                    coszenithp=(np.sin(latitude)*np.cos(lon[l])*np.sin(zenith[i,j])*np.cos(azimuth[i,j])+np.sin(latitude)*np.sin(lon[l])*np.sin(zenith[i,j])*np.sin(azimuth[i,j])+np.cos(zenith[i,j])*np.cos(latitude))
                    zenithp=np.arccos(coszenithp)
                    cosazimuthp=(np.cos(latitude)*np.cos(lon[l])*np.sin(zenith[i,j])*np.cos(azimuth[i,j]) +np.sin(lon[l])*np.cos(latitude)*np.sin(zenith[i,j])*np.sin(azimuth[i,j]) -np.sin(latitude)*np.cos(zenith[i,j]) ) / np.sin(zenithp)
                    sinazimuthp=(-np.sin(lon[l])*np.sin(zenith[i,j])*np.cos(azimuth[i,j])+ np.cos(lon[l])*np.sin(zenith[i,j])*np.sin(azimuth[i,j])) / np.sin(zenithp)
                    if zenithp==0:
                        print('zenithp=0')
                        print(zenith[i,j],azimuth[i,j])
                        print(zenithp,cosazimuthp,sinazimuthp)
                    if cosazimuthp<-1.1 or cosazimuthp>1.1:
                        print('cos out of range')
                        print(zenith[i,j],azimuth[i,j])
                        print(zenithp,cosazimuthp,sinazimuthp)
        
                    if sinazimuthp<0:
                        if cosazimuthp<-1:
                            azimuthp=2*np.pi-np.arccos(-1)
                        elif cosazimuthp>1:
                            azimuthp=2*np.pi-np.arccos(1)       
                        else:
                            azimuthp=2*np.pi-np.arccos(cosazimuthp)

                    else:
                        if cosazimuthp<-1:
                            azimuthp=np.arccos(-1)
                        elif cosazimuthp>1:
                            azimuthp=np.arccos(1)    
                        else:
                            azimuthp=np.arccos(cosazimuthp)
                    diffzenith=zenithp-zenith[i,j]
                    diffazimuth=azimuthp-azimuth[i,j]

                    ip=int(i+round(diffazimuth/(dphi*np.pi/180)))
                    jp=int(j+round(diffzenith/(dtheta*np.pi/180)))
                
                    contribX = (lefftX[f,ip,jp]**2+leffpX[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j]) 
                    contribY = (lefftY[f,ip,jp]**2+leffpY[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j]) 
                    contribZ = (lefftZ[f,ip,jp]**2+leffpZ[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j])  
                    contribXX = abs(RFchainNS[f])*abs(RFchainNS[f])*(lefftX[f,ip,jp]**2+leffpX[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j]) 
                    contribYY = abs(RFchainEW[f])*abs(RFchainEW[f])*(lefftY[f,ip,jp]**2+leffpY[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j]) 
                    contribZZ = abs(RFchainZ[f])*abs(RFchainZ[f])*(lefftZ[f,ip,jp]**2+leffpZ[f,ip,jp]**2)*temp[i,j]*np.sin(zenith[i,j]) 
                
                    #print(contrib)
                    if contribX!=0:
                        T[f,l,i,j]=temp[i,j]
                    #omega=omega+np.sin(zenith[i,j])
                    voc2X[f,l]=voc2X[f,l] + contribX
                    voc2Y[f,l]=voc2Y[f,l] + contribY
                    voc2Z[f,l]=voc2Z[f,l] + contribZ
                
                    vout2X[f,l]=vout2X[f,l] + contribXX
                    vout2Y[f,l]=vout2Y[f,l] + contribYY
                    vout2Z[f,l]=vout2Z[f,l] + contribZZ
                
                    if l==0:
                        avleff2X[f]=avleff2X[f] + (lefftX[f,i,j]**2+leffpX[f,i,j]**2)*np.sin(zenith[i,j])
                        avleff2Y[f]=avleff2Y[f] + (lefftY[f,i,j]**2+leffpY[f,i,j]**2)*np.sin(zenith[i,j])
                        avleff2Z[f]=avleff2Z[f] + (lefftZ[f,i,j]**2+leffpZ[f,i,j]**2)*np.sin(zenith[i,j])
                        avtemp[f]=avtemp[f] + temp[i,j]*np.sin(zenith[i,j]) 
                        avBnu[f]=avBnu[f] + temp[i,j]*2*(freqs[f]*1e6)**2*kB/(c**2)*np.sin(zenith[i,j]) 
                        avErms2[f]=avErms2[f] + temp[i,j]*4*(np.pi)*Z0*(freqs[f]*1e6)**2*kB/(c**2)*np.sin(zenith[i,j]) 
                    #lon0Voc2[f]= lon0Voc2[f]+ (lefft[f,ip,jp]**2+leffp[f,ip,jp]**2)* Z0* temp[i,j] * (freqs[f]*1e6)**2 * kB/(c**2) *np.sin(zenith[i,j]) 
        
        voc2X[f,:]=voc2X[f,:]*(freqs[f]*1e6)**2
        voc2Y[f,:]=voc2Y[f,:]*(freqs[f]*1e6)**2
        voc2Z[f,:]=voc2Z[f,:]*(freqs[f]*1e6)**2
        vout2X[f,:]=vout2X[f,:]*(freqs[f]*1e6)**2
        vout2Y[f,:]=vout2Y[f,:]*(freqs[f]*1e6)**2
        vout2Z[f,:]=vout2Z[f,:]*(freqs[f]*1e6)**2
    voc2X=voc2X*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz
    voc2Y=voc2Y*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz
    voc2Z=voc2Z*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz

    vout2X=vout2X*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz
    vout2Y=vout2Y*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz
    vout2Z=vout2Z*kB*Z0/(c**2)*dtheta*np.pi/180*dphi*np.pi/180 #per Hz

    avleff2X=avleff2X*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)
    avleff2Y=avleff2Y*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)
    avleff2Z=avleff2Z*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)
    avtemp=avtemp*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)
    avBnu=avBnu*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)
    avErms2=0.5*avErms2*dtheta*np.pi/180*dphi*np.pi/180/(4*np.pi)

    filename = grand_add_path_data("detector/RFchain_v2/Z_ant_3.2m.csv")
    RLX,RLY,RLZ = np.loadtxt(filename, delimiter=",", usecols = (1,3,5), skiprows=1, unpack=True)
    RLbisX=np.zeros((len(freqs),len(lon)))
    RLbisY=np.zeros((len(freqs),len(lon)))
    RLbisZ=np.zeros((len(freqs),len(lon)))
    RLbisX=RLbisX.T
    RLbisY=RLbisY.T
    RLbisZ=RLbisZ.T
    RLbisX[:]=RLX
    RLbisY[:]=RLY
    RLbisZ[:]=RLZ
    RLbisX=RLbisX.T
    RLbisY=RLbisY.T
    RLbisZ=RLbisZ.T
    plX=voc2X/(4*RLbisX)
    plY=voc2Y/(4*RLbisY)
    plZ=voc2Z/(4*RLbisZ)
    
    # Saving and plotting based on run_mode
    if run_mode == "Voc":
        np.save(f"galactic_Voc2_per_Hz_{site}_{du_type}.npy", np.stack([voc2X, voc2Y, voc2Z], axis=-1))
        
        plt.figure(figsize=(12, 8))
        plt.title('Square of Open Circuit Voltage vs LST', fontsize=20)
        plt.plot(lstaxis, np.sum(voc2X, 0) * dnu * 1e6, 'k')
        plt.plot(lstaxis, np.sum(voc2Y, 0) * dnu * 1e6, 'y')
        plt.plot(lstaxis, np.sum(voc2Z, 0) * dnu * 1e6, 'b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('LST [h]', fontsize=16)
        plt.ylabel('V$_{oc,RMS}^2$ [V$^2$]', fontsize=16)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_Voc2_LST_{site}_{du_type}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.title(f'Square of open circuit Voltage per Hz vs Frequency at LST {lst_value} h', fontsize=20)
        plt.plot(freqs, voc2X[:, lst_value == lstaxis].flatten(), '-*', color='k')
        plt.plot(freqs, voc2Y[:, lst_value == lstaxis].flatten(), '-*', color='y')
        plt.plot(freqs, voc2Z[:, lst_value == lstaxis].flatten(), '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('V$_{oc,RMS}^2$ [V$^2$/Hz]')
        plt.savefig(f"galactic_Voc2_freq_{site}_{du_type}.png")
        plt.close()

    elif run_mode == "Vout":
        np.save(f"galactic_Vout2_per_Hz_{site}_{du_type}.npy", np.stack([vout2X, vout2Y, vout2Z], axis=-1))
        
        plt.figure(figsize=(12, 8))
        plt.title('Square of Output Voltage vs LST', fontsize=20)
        plt.plot(lstaxis, np.sum(vout2X, 0) * dnu * 1e6, 'k')
        plt.plot(lstaxis, np.sum(vout2Y, 0) * dnu * 1e6, 'y')
        plt.plot(lstaxis, np.sum(vout2Z, 0) * dnu * 1e6, 'b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('LST [h]', fontsize=16)
        plt.ylabel('V$_{out,RMS}^2$ [V$^2$]', fontsize=16)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_Vout2_LST_{site}_{du_type}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.title(f'Square of Output Voltage per Hz vs Frequency at LST {lst_value} h', fontsize=20)
        plt.plot(freqs, vout2X[:, lst_value == lstaxis].flatten(), '-*', color='k')
        plt.plot(freqs, vout2Y[:, lst_value == lstaxis].flatten(), '-*', color='y')
        plt.plot(freqs, vout2Z[:, lst_value == lstaxis].flatten(), '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('V$_{out,RMS}^2$ [V$^2$/Hz]')
        plt.savefig(f"galactic_Vout2_freq_{site}_{du_type}.png")
        plt.close()

    elif run_mode == "PL":
        np.save(f"galactic_PL_per_Hz_{site}_{du_type}.npy", np.stack([plX, plY, plZ], axis=-1))
        
        plt.figure(figsize=(12, 8))
        plt.title('Galactic Power vs LST', fontsize=20)
        plt.plot(lstaxis, np.sum(plX, 0) * dnu * 1e6 / 4, 'k')
        plt.plot(lstaxis, np.sum(plY, 0) * dnu * 1e6 / 4, 'y')
        plt.plot(lstaxis, np.sum(plZ, 0) * dnu * 1e6 / 4, 'b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('LST [h]', fontsize=16)
        plt.ylabel('P$_L$ [W]', fontsize=16)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_PL_LST_{site}_{du_type}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.title(f'Galactic Power per Hz vs Frequency at LST {lst_value}h', fontsize=20)
        plt.plot(freqs, plX[:, lst_value == lstaxis].flatten(), '-*', color='k')
        plt.plot(freqs, plY[:, lst_value == lstaxis].flatten(), '-*', color='y')
        plt.plot(freqs, plZ[:, lst_value == lstaxis].flatten(), '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.legend(["port X", "port Y", "port Z"], loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('P$_L$ [W/Hz]')
        plt.savefig(f"galactic_PL_freq_{site}_{du_type}.png")
        plt.close()

    elif run_mode == "Efield":
        np.save(f"galactic_Efield2_per_Hz_{site}_{du_type}.npy", np.stack([avtemp, avBnu, avErms2], axis=-1))

        plt.figure(figsize=(12, 8))
        plt.title('Average Galactic Temperature vs Frequency', fontsize=20)
        plt.plot(freqs, avtemp * kB, '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('Frequency [MHz]', fontsize=16)
        plt.ylabel('k$_B$<T>$_{4\\pi}$ [J]', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_temp_{site}_{du_type}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.title('Average Galactic Radiance vs Frequency', fontsize=20)
        plt.plot(freqs, avBnu, '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('Frequency [MHz]', fontsize=16)
        plt.ylabel('<B$_{v}$>$_{4\\pi}$ [W$\cdot$m$^{-2}$$\cdot$sr$^{-1}$$\cdot$Hz$^{-1}$]', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_bnu_{site}_{du_type}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.title('Average Galactic Square Electric Field vs Frequency', fontsize=20)
        plt.plot(freqs, avErms2, '-*', color='b')
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel('Frequency [MHz]', fontsize=16)
        plt.ylabel('<|E|$^{2}$$_{rms}$>$_{4\\pi}$ [V$^{2}$/m$^{2}$$\cdot$Hz$^{-1}$]', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"galactic_avErms2_{site}_{du_type}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Galactic Noise Data")
    parser.add_argument("--site", type=str, choices=["gp13", "gaa"], default="gp13", help="Site location")
    parser.add_argument("--du_type", type=str, choices=["GP300", "GP300_nec", "GP300_mat"], default="GP300", help="Detector unit type")
    parser.add_argument("--run_mode", type=str, choices=["Voc", "Vout", "PL", "Efield"], required=True, help="Run mode")
    parser.add_argument("--lst_value", type=int, default=18, choices=range(0, 24), help="LST value for frequency plot")

    args = parser.parse_args()
    main(args.site, args.du_type, args.run_mode, args.lst_value)