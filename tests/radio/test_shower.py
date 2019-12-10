# -*- coding: utf-8 -*-
"""
Unit tests for the grand.radio.shower module

Usage: python -m tests.radio.test_shower

"""

import unittest
import sys

import astropy.units as u
from grand.radio.shower import Shower, SimulatedShower, ReconstructedShower
import numpy as np


class ShowerTest(unittest.TestCase):
    """Unit tests for the version module"""

    #def _init_(self): # Constructor
    shower1 = Shower()


    def test_showerID(self):
        ID = []
        for i in range(1): #(2):
            shower_ID=None
            ID.append("name"+str(i))
            # add ID to shower ---  can only get one ID per shower
            self.shower1.showerID = str("name"+str(i))
            # get ID from shower
            shower_ID = self.shower1.showerID

            if i==0:
                self.assertEqual(ID[0], shower_ID) #shall be equal
                self.assertEqual(len(ID[0]), len(shower_ID))


    def test_primary(self):
        prim = []

        prim.append("proton")
        # add primary to shower ---  can only get one primary element per shower
        self.shower1.primary = prim[-1]
        # get primary from shower
        primary = self.shower1.primary
        self.assertEqual(prim[0], primary) #shall be equal


    def test_energy(self):
        ener = []

        ener.append(1e18* u.eV )
        # add energy to shower ---  can only get one primary element per shower
        self.shower1.energy = ener[-1]
        # get energy from shower
        energy = self.shower1.energy
        self.assertEqual(ener[0], energy) #shall be equal


    def test_zenith(self):
        zen = []

        zen.append(70.* u.deg)
        # add zenithto shower ---  can only get one primary element per shower
        self.shower1.zenith = zen[-1]
        # get zenith from shower
        zenith = self.shower1.zenith
        self.assertEqual(zen[0], zenith) #shall be equal


    def test_azimuth(self):
        azim = []

        azim.append(100.* u.deg)
        # add azimuthto shower ---  can only get one primary element per shower
        self.shower1.azimuth = azim[-1]
        # get azimuth from shower
        azimuth = self.shower1.azimuth
        self.assertEqual(azim[0], azimuth) #shall be equal


    def test_injectionheight(self):
        injh = []

        injh.append(1500.* u.m)
        # add injectionheightto shower ---  can only get one primary element per shower
        self.shower1.injectionheight = injh[-1]
        # get injectionheight from shower
        injectionheight = self.shower1.injectionheight
        self.assertEqual(injh[0], injectionheight) #shall be equal


    def test_trigger(self):
        tr = []

        tr.append((1, "yes", 45* u.mV))
        # add injectionheightto shower ---  can only get one primary element per shower
        self.shower1.trigger = tr[-1]
        # get trigger from shower
        trigger = self.shower1.trigger
        self.assertEqual(tr[0], trigger) #shall be equal


    shower2 = Shower(azimuth = 60.* u.deg, zenith = 45.* u.deg)

    def test_direction(self):
        azi = self.shower2.azimuth
        zeni = self.shower2.zenith
        dire = np.array([np.cos(azi)*np.sin(zeni),
                             np.sin(azi)*np.sin(zeni),
                             np.cos(zeni)])

        # get direction from shower
        direction = self.shower2.direction()

        self.assertEqual(dire.all(), direction.all()) #shall be equal


class Sim_ShowerTest(unittest.TestCase):

    shower3 = SimulatedShower()

    def test_sim_shower(self):
        # add simulation to shower
        self.shower3.simulation = "coreas"
        # get simulation
        sim = self.shower3.simulation

        self.assertEqual("coreas", sim)

        # add Xmax to shower
        self.shower3.Xmax = 600. *u.g/(u.cm**2)
        # get Xmax
        Xmax = self.shower3.Xmax

        self.assertEqual(600. *u.g/(u.cm**2), Xmax)


class Reco_ShowerTest(unittest.TestCase):

    shower4 = ReconstructedShower()

    def test_recoenergy(self):
        recoener = []

        recoener.append(1e18* u.eV )
        # add energy to shower ---  can only get one primary element per shower
        self.shower4.recoenergy = recoener[-1]
        # get energy from shower
        recoenergy = self.shower4.recoenergy
        self.assertEqual(recoener[0], recoenergy) #shall be equal


    def test_recozenith(self):
        recozen = []

        recozen.append(70.* u.deg)
        # add zenithto shower ---  can only get one primary element per shower
        self.shower4.recozenith = recozen[-1]
        # get zenith from shower
        recozenith = self.shower4.recozenith
        self.assertEqual(recozen[0], recozenith) #shall be equal


    def test_recoazimuth(self):
        recoazim = []

        recoazim.append(100.* u.deg)
        # add azimuthto shower ---  can only get one primary element per shower
        self.shower4.recoazimuth = recoazim[-1]
        # get azimuth from shower
        recoazimuth = self.shower4.recoazimuth
        self.assertEqual(recoazim[0], recoazimuth) #shall be equal


    def test_recoXmax(self):
        # add Xmax to shower
        self.shower4.recoXmax = 600. *u.g/(u.cm**2)
        # get Xmax
        recoXmax = self.shower4.recoXmax

        self.assertEqual(600. *u.g/(u.cm**2), recoXmax)


    shower5 = ReconstructedShower(recoazimuth = 60.* u.deg, recozenith = 45.* u.deg)

    def test_recodirection(self):
        azi = self.shower5.recoazimuth
        zeni = self.shower5.recozenith
        dire = np.array([np.cos(azi)*np.sin(zeni),
                             np.sin(azi)*np.sin(zeni),
                             np.cos(zeni)])

        # get direction from shower
        direction = self.shower5.recodirection()

        self.assertEqual(dire.all(), direction.all()) #shall be equal


if __name__ == "__main__":
    unittest.main()
