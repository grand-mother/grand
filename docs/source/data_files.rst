Data files
==========

.. contents::
   :local:
   :depth: 2

GRANDlib relies on about 1 GB of tabulated models: antenna effective lengths,
RF-chain measurements, Galactic-noise tables, terrain and the geomagnetic
field.  Most of it is downloaded by ``env/setup.sh`` rather than kept in
version control.  This page describes each file, where it comes from and what
reads it.

What is in version control
--------------------------

``data/.gitignore`` ignores everything and re-admits a handful of files, so a
fresh checkout has 14 tracked files and no models:

==================================  ========================================
Tracked                             What it is
==================================  ========================================
``download_*.py`` (4 scripts)       Fetch the models; see below
``egm96.png``                       EGM96 geoid undulation map, ~2 MB.  Read
                                    by :func:`~grand.geo.topography.geoid_undulation`.
                                    A PNG used as a raster data file, not a
                                    picture
``geomagnet/IGRF13.COF``            IGRF-13 geomagnetic coefficients
``geomagnet/WMM2020.COF``           World Magnetic Model 2020 coefficients
``model_version.flag``              Which model release the download script
                                    should fetch: ``444342 20250313``
``noise/galactic_PL_*.npy`` (3)     The Galactic-noise tables in use, one
                                    per antenna model (see below)
``map.png``, ``readme.md``          Documentation assets
==================================  ========================================

Everything else is downloaded.  After ``source env/setup.sh`` a working tree
holds roughly:

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Directory
     - Size
     - Contents
   * - ``data/detector/``
     - ~999 MB
     - Antenna effective lengths, RF-chain S-parameters
   * - ``data/noise/``
     - ~21 MB
     - Galactic-noise tables and the LFMap sky maps
   * - ``data/topography/``
     - varies
     - SRTM elevation tiles, one per one-degree square
   * - ``data/geomagnet/``
     - ~160 kB
     - The two coefficient files above

.. note::

   ``data/test_efield.root`` appears on many developer machines and is **not**
   tracked or downloaded by anything.  No test reads it any more: they use the
   committed samples under ``sim2root/Common`` and write into temporary
   folders, never into ``data/``.

Checking the installation
-------------------------

``data/download_data_grand.py`` records the files it installs, with their
sizes and SHA-256 sums, in ``data/data_model_manifest.json``.  The antenna,
RF-chain and noise loaders check a file against it before reading: a missing
file, or one whose size differs, stops with a message naming the file and the
remedy, rather than a library traceback or, for a damaged file that still
parses, silently different voltages.  The downloader re-downloads when the
version matches but a directory or file is missing.  To check by hand, sums
included::

    python -m grand.basis.data_model            # verify
    python -m grand.basis.data_model --write    # record the current files

An installation from before the manifest gets one the next time the
downloader runs.

The download scripts
--------------------

Four scripts, four archives.

============================================  =================================
Script                                        Fetches
============================================  =================================
``download_data_grand.py``                    ``grand_model_<version>.tar.gz``,
                                              the version taken from
                                              ``model_version.flag``
``download_grand_antenna_models.py``          ``grand_model_20241218.tar.gz``
``download_LFmap_grand.py``                   ``LFmap.tar.gz``
``download_new_RFchain.py``                   ``RF_chain_20241218.tar.gz``
============================================  =================================

All of them fetch from ``forge.in2p3.fr``, which requires
no credentials but is not a mirror-backed host: if it is down, a fresh
environment cannot be built.

``env/setup.sh`` runs only ``download_data_grand.py``; the line for
``download_new_RFchain.py`` is commented out.  So the versioned bundle is what
a normal setup gets, and the other three archives are fetched by hand when
someone needs them.

Antenna effective length
------------------------

``data/detector/Light_GP300Antenna_*_leff.npz``, nine files: three arms for
each of three antenna simulations.

===================================  =======  =====================
File                                 Model    Arm
===================================  =======  =====================
``..._EWarm_leff.npz``               HFSS     east-west
``..._SNarm_leff.npz``               HFSS     south-north
``..._Zarm_leff.npz``                HFSS     vertical
``..._nec_Xarm_leff.npz``            NEC      south-north
``..._nec_Yarm_leff.npz``            NEC      east-west
``..._nec_Zarm_leff.npz``            NEC      vertical
``..._mat_Xarm_leff.npz``            MATLAB   south-north
``..._mat_Yarm_leff.npz``            MATLAB   east-west
``..._mat_Zarm_leff.npz``            MATLAB   vertical
===================================  =======  =====================

.. warning::

   The arm column above comes from correlating each pattern with the named
   HFSS arms, not from the file names.  **X is the south-north arm and Y is
   the east-west arm**, the opposite of what the GRANDlib Handbook says.  See :ref:`issue-handbook-arm-naming`; the
   measurement is in ``tests/sim/test_antenna_arm_identity.py``.

Each archive holds ``freq_mhz`` (221 bins, 30–250 MHz) and the complex
``leff_theta`` and ``leff_phi``, each of shape ``(361, 91, 221)`` indexed by
azimuth, zenith and frequency.

Two details matter when reading them.  :class:`~grand.sim.detector.antenna_model.AntennaModel`
transposes on load, so the in-memory arrays are indexed ``[frequency, azimuth,
zenith]`` and are named ``leff_theta_reim``; the attributes ``leff_theta`` and
``phase_theta`` exist on the loaded object but are ``None``, because the polar
form is never populated.  And the in-memory frequency axis is in **hertz**,
while everything in :mod:`grand.sim.detector.rf_chain` uses megahertz
(:ref:`quickstart-units`).

Notebook 03 works through all of this.

RF-chain S-parameters
---------------------

``data/detector/RFchain_v2/`` holds 33 files.  The default
:class:`~grand.sim.detector.rf_chain.RFChain` reads seven of them, two per
axis:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Stage attribute
     - Class
     - File (X axis)
   * - ``matcnet``
     - ``MatchingNetwork``
     - ``MatchingNetworkX.s2p``
   * - ``lna``
     - ``LowNoiseAmplifier``
     - ``LNA-X.s2p``
   * - ``balun1``
     - ``BalunAfterLNA``
     - ``balun_in_nut.s2p``
   * - ``cable``
     - ``Cable``
     - ``cable+Connector.s2p``
   * - ``vgaf``
     - ``VGAFilter``
     - ``feb+amfitler+biast.s2p``
   * - ``balun2``
     - ``BalunBeforeADC``
     - ``balun_before_ad.s2p``
   * - ``zload``
     - ``Zload``
     - ``S_balun_AD.s1p``

The ``antenna_LNA_{X,Y,Z}_frontend0db.s2p`` files belong to
``gaa_frontend0db``, which the default chain does not instantiate; they are
reached only through the GAA variant.

Six files are read by no chain at all:

- ``filter+vga0db+filter.s2p``
- ``filter+vga5db+filter.s2p``
- ``filter+vga20db+filter.s2p``
- ``High_freq_pass_PAnalogfilter.s2p``
- ``balun13in20230612.s2p``
- ``zload_balun_200ohm.s1p``

The first three are the :term:`variable-gain amplifier <VGA>` tables; that no chain reads
them is :ref:`issue-vga-gain-ignored`.  The stage named for the VGA,
``vgaf``, loads ``feb+amfitler+biast.s2p``, a front-end board with an AM
filter and a bias tee.

Which files the chain reads is set by an XML component configuration parsed at
import; ``grand.sim.detector.rf_chain.components`` holds the result and is the
quickest way to see the current mapping.

Galactic noise
--------------

The tables in use are ``noise/galactic_PL_per_Hz_gp13_GP300.npy`` and its
``_nec`` and ``_mat`` counterparts, one per antenna model.  Each has shape
``(221, 72, 3)``: 30 to 250 MHz in 1 MHz steps, 72 local-sidereal-time bins
of 20 minutes, and the three arms.  They hold the **available power spectral
density** :math:`P_L`, in W/Hz.

They were computed by ``grand/sim/noise/Compute_Plot_Galactic_Noise.py``,
which integrates the :term:`LFMap` sky temperature over direction against the
antenna's :term:`effective length`, :math:`|\ell_\theta|^2 + |\ell_\phi|^2`, and
converts the resulting :term:`open-circuit voltage` to available power:

.. math::  P_L = \frac{V_{\rm oc,RMS}^2}{4\,\mathrm{Re}(Z_{\rm ant})}

with :math:`Z_{\rm ant}` from ``detector/RFchain_v2/Z_ant_3.2m.csv``.  The
LFMap inputs, ``noise/LFmap/LFmapshort<frequency>.npy``, are fetched by
``data/download_LFmap_grand.py``.  The three models differ only in the
effective-length files they read:

=============  ===================================================
``du_type``    Effective length
=============  ===================================================
``GP300``      ``Light_GP300Antenna_{SNarm,EWarm,Zarm}_leff.npz``
``GP300_nec``  ``Light_GP300Antenna_nec_{X,Y,Z}arm_leff.npz``
``GP300_mat``  ``Light_GP300Antenna_mat_{X,Y,Z}arm_leff.npz``
=============  ===================================================

:func:`~grand.sim.noise.galaxy.galactic_noise` inverts the last step,
:math:`V_{\rm oc,RMS}^2 = 4 P_L \mathrm{Re}(Z_{\rm ant})`, and
``tests/sim/test_galactic_noise_normalisation.py`` checks the simulated level
against the same relation, computed independently.

**Older tables.**  ``Vocmax_30-250MHz_uVperMHz_{hfss,nec,mat}.npy``, with the
matching ``Pocmax_`` and ``Voutmax_`` sets and ``PG_ALL_jifen.mat``, are the
tables used before 7 September 2026.  They are still shipped, to describe
files simulated before then, but no code reads them.  The ``_nec`` and
``_mat`` files of that set are identical, and the default model read
``PG_ALL_jifen.mat`` instead, so the three models gave two sets of numbers
differing by up to a factor of two (:ref:`issue-galactic-noise-tables`).

Topography
----------

``data/topography/`` holds :term:`SRTM` tiles, one ``.hgt`` per one-degree square,
named after the south-west corner: ``N41E096.hgt`` covers 41–42 °N, 96–97 °E.
They are a few megabytes each and are **not** in version control, so a fresh
checkout has none.

:func:`grand.geo.topography.update_data` downloads what a region needs.  A
lookup with no tile returns ``nan`` rather than raising; see
:doc:`troubleshooting`.

Geomagnetic field
-----------------

``data/geomagnet/`` holds the two coefficient files, and both **are** in version
control, so :mod:`grand.geo.geomagnet` works on a fresh checkout.

``IGRF13.COF`` is IGRF-13, whose published validity ended on 1 January 2025.
It still evaluates outside that window, silently.  IGRF-14 was released in
2024 and has not been adopted here; see :doc:`known_issues`.
