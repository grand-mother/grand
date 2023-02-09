'''
Created on 8 févr. 2023

@author: jcolley
'''
from grand.simu.du.model_ant_du import AntennaModelGp300
import numpy as np


def create_small_leff_file():
    agp = AntennaModelGp300()
    agp.set_antenna_model("GP300Antenna")
    file_out = 'Light_GP300Antenna_EWarm_leff'
    leff_phi_cart = agp.leff_ew.table.leff_phi_cart[:,:91,:]
    leff_theta_cart = agp.leff_ew.table.leff_theta_cart[:,:91,:]
    freq = agp.leff_ew.table.frequency/1e6
    # pas possible de def un dictionnaire
    #m_dict = {"leff_phi": leff_phi_cart, "leff_theta":leff_ew_cart}
    #np.savez(file_out, m_dict)
    history=np.array(['From file GP300Antenna_EWarm_leff.npy, leff converted in complex number, modification of axis freq is now the last.'])
    comment=np.array(['shape  leff_xx (phi=361, theta=91, freq=221): theta is colatitude 0 to 90 deg, phi is plan direction 0 to 360 deg'])
    version=np.array(['1.0'])
    author=np.array(['Colley Jean-Marc, jcolley@lpnhe.in2p3.fr'])
    np.savez(file_out, leff_phi=leff_phi_cart, leff_theta=leff_theta_cart, freq_mhz=freq,
             history=history,
             comment=comment, 
             version=version,
             author=author)
    file_out = 'Light_GP300Antenna_SNarm_leff'
    leff_phi_cart = agp.leff_sn.table.leff_phi_cart[:,:91,:]
    leff_theta_cart = agp.leff_sn.table.leff_theta_cart[:,:91,:]
    # pas possible de def un dictionnaire
    #m_dict = {"leff_phi": leff_phi_cart, "leff_theta":leff_ew_cart}
    #np.savez(file_out, m_dict)
    history=np.array(['From file GP300Antenna_SNarm_leff.npy, leff converted in complex number, modification of axis freq is now the last.'])
    np.savez(file_out, leff_phi=leff_phi_cart, leff_theta=leff_theta_cart, freq_mhz=freq,
             history=history,
             comment=comment, 
             version=version,
             author=author)
    file_out = 'Light_GP300Antenna_Zarm_leff'
    leff_phi_cart = agp.leff_z.table.leff_phi_cart[:,:91,:]
    leff_theta_cart = agp.leff_z.table.leff_theta_cart[:,:91,:]
    # pas possible de def un dictionnaire
    #m_dict = {"leff_phi": leff_phi_cart, "leff_theta":leff_ew_cart}
    #np.savez(file_out, m_dict)
    history=np.array(['From file GP300Antenna_Zarm_leff.npy, leff converted in complex number, modification of axis freq is now the last.'])
    np.savez(file_out, leff_phi=leff_phi_cart, leff_theta=leff_theta_cart, freq_mhz=freq,
             history=history,
             comment=comment, 
             version=version,
             author=author)


if __name__ == '__main__':
    create_small_leff_file()
