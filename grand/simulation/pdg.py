from enum import IntEnum

__all__ = ['ParticleCode']


class ParticleCode(IntEnum):
    '''PDG Monte Carlo particle numbering scheme

       Ref: http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    '''

    # Bosons
    GAMMA = 22
    Z_0 = 23
    W_PLUS = 24
    W_MINUS = -24

    # Leptons
    ELECTRON = 11
    ANTI_ELECTRON = -11
    NEUTRINO_E = 12
    ANTI_NEUTRINO_E = -12
    MUON = 13
    ANTI_MUON = -13
    NEUTRINO_MU = 14
    ANTI_NEUTRINO_MU = -14
    TAU = 15
    ANTI_TAU = -15
    NEUTRINO_TAU = 16
    ANTI_NEUTRINO_TAU = -16

    # Mesons
    PION_0 = 111
    PION_PLUS = 211
    PION_MINUS = -211

    # Baryons
    PROTON = 2212
    NEUTRON = 2112

    # Atoms
    IRON = 1000260560
