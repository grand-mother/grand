from enum import IntEnum

__all__ = ["ParticleCode"]


class ParticleCode(IntEnum):
    """PDG Monte Carlo particle numbering scheme

    Ref: http://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf
    """

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

    @classmethod
    def _missing_(cls, value):
        r"""Accepts a member name, case-insensitively; otherwise lists the valid codes.

        ``ParticleCode('proton')`` raised "'proton' is not a valid
        ParticleCode", without saying what is (#267).
        """
        if isinstance(value, str):
            member = cls.__members__.get(value.strip().upper().replace("-", "_").replace(" ", "_"))
            if member is not None:
                return member
        raise ValueError("GRANDlib: ParticleCode: %r is not a particle; give a PDG code or one of %s"
                         % (value, ", ".join("%s (%d)" % (name, int(code))
                                             for name, code in cls.__members__.items())))
