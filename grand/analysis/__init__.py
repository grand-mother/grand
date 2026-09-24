"""Reconstruction of shower parameters from recorded times and amplitudes.

From ``dev_marion`` (Marion Guelfand, 2026); see README.md here.  The
subpackages are exposed under the short names the examples use.
"""

import grand.analysis.signals as sig
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.energy_reco as en
import grand.analysis.coords.array_shower as co
import grand.analysis.geom as geom

__all__ = ['sig', 'fit', 'cons', 'en', 'co', 'geom']
