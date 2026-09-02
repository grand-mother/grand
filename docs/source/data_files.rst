Data files
==========

.. contents::
   :local:
   :depth: 2

GRANDlib is about a gigabyte of tabulated measurements with a Python package
around it.  Almost none of that is in version control, and nothing else in this
documentation says what any of it is.  This page does.

Everything here was checked against the files on disk rather than taken from
the download scripts or the Handbook, both of which are out of date in places.

What is in version control
--------------------------

``data/.gitignore`` ignores everything and re-admits a handful of files, so a
fresh checkout has 12 tracked files and no models:

==================================  ========================================
Tracked                             What it is
==================================  ========================================
``download_*.py`` (5 scripts)       Fetch the models; see below
``egm96.png``                       EGM96 geoid undulation map, ~2 MB.  Read
                                    by :func:`~grand.geo.topography.geoid_undulation`.
                                    A PNG used as a raster data file, not a
                                    picture
``geomagnet/IGRF13.COF``            IGRF-13 geomagnetic coefficients
``geomagnet/WMM2020.COF``           World Magnetic Model 2020 coefficients
``model_version.flag``              Which model release the download script
                                    should fetch: ``444342 20250313``
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
   tracked or downloaded by anything.  A test that needs it is marked xfail for
   that reason; see :doc:`testing`.

The download scripts
--------------------

Five scripts, four distinct archives, and one exact duplicate.

============================================  =================================
Script                                        Fetches
============================================  =================================
``download_data_grand.py``                    ``grand_model_<version>.tar.gz``,
                                              the version taken from
                                              ``model_version.flag``
``download_grand_antenna_models.py``          ``grand_model_20241218.tar.gz``
``download_LFmap_grand.py``                   ``LFmap.tar.gz``
``download_new_RFchain.py``                   ``RF_chain_20241218.tar.gz``
``download_new_RFchain_grand.py``             the same archive
============================================  =================================

The last two differ **only by a trailing newline** — they are the same file
committed twice.  All of them fetch from ``forge.in2p3.fr``, which requires
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

   The X/Y column above is measured, not read off the file names.  **X is the
   south-north arm and Y is the east-west arm**, which is the opposite of what
   the GRANDlib Handbook says.  See :ref:`issue-handbook-arm-naming`; the
   measurement is in ``tests/sim/test_antenna_arm_identity.py``.

Each archive holds ``freq_mhz`` (221 bins, 30–250 MHz) and the complex
``leff_theta`` and ``leff_phi``, each of shape ``(361, 91, 221)`` indexed by
azimuth, zenith and frequency.

Two things catch people out.  :class:`~grand.sim.detector.antenna_model.AntennaModel`
transposes on load, so the in-memory arrays are indexed ``[frequency, azimuth,
zenith]`` and are named ``leff_theta_reim``; the attributes ``leff_theta`` and
``phase_theta`` exist on the loaded object but are ``None``, because the polar
form is never populated.  And the in-memory frequency axis is in **hertz**,
while everything in :mod:`grand.sim.detector.rf_chain` uses megahertz — see
:doc:`implementation`.

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

The first three are the variable-gain amplifier tables, and their absence from
the read set is the whole of :ref:`issue-vga-gain-ignored`.  Note the ``vgaf``
row above: the stage named for the VGA loads ``feb+amfitler+biast.s2p``, a
front-end board with an AM filter and a bias tee.  It is not reading the wrong
VGA table; it is reading a different component.

Which files the chain reads is set by an XML component configuration parsed at
import; ``grand.sim.detector.rf_chain.components`` holds the result and is the
quickest way to see the current mapping.

Galactic noise
--------------

``data/noise/`` holds two generations of the model.

**The tables.**  ``Vocmax_30-250MHz_uVperMHz_{hfss,nec,mat}.npy``, with
matching ``Pocmax_...`` (power) and ``Voutmax_...`` (after the chain) sets.
Each is shape ``(221, 24, 3)`` — frequency, LST hour, arm — and
:func:`~grand.sim.noise.galaxy.galactic_noise` transposes to
``(frequency, arm, LST)``.

**The source data.**  ``PG_ALL_jifen.mat``, integrated sky power, and
``LFmap/``, the LFMap sky maps the tables were built from.

Three problems, all in :ref:`issue-galactic-noise-tables`:

- the ``_nec`` and ``_mat`` files are **byte-identical**, so ``du_type='GP300_nec'``
  and ``'GP300_mat'`` select the same numbers;
- the default ``du_type='GP300'`` reads none of the ``.npy`` tables — it
  recomputes :math:`V_{\rm oc}^2 = 4 P R_{\rm ant}` from ``PG_ALL_jifen.mat``;
- the ``_hfss`` tables, the highest-level ones shipped, are opened by nothing.

Band-integrated level at LST 18 h, per arm, in microvolts:

=====================  =========================  =====================
Selector               Reads                      X, Y, Z
=====================  =========================  =====================
``GP300`` (default)    ``PG_ALL_jifen.mat``       27.8, 35.6, 31.4
``GP300_nec``          ``Vocmax_..._nec.npy``     44.5, 55.6, 53.4
``GP300_mat``          the same file as ``nec``   44.5, 55.6, 53.4
*(unreachable)*        ``Vocmax_..._hfss.npy``    59.6, 75.5, 66.9
=====================  =========================  =====================

**Quote the** ``du_type`` **with any absolute noise level.**  Without it the
number is ambiguous by a factor of two, quite apart from the separate
:math:`\sqrt2` question of :ref:`issue-galactic-noise-normalisation`.

Topography
----------

``data/topography/`` holds SRTM tiles, one ``.hgt`` per one-degree square,
named after the south-west corner: ``N41E096.hgt`` covers 41–42 °N, 96–97 °E.
They are a few megabytes each and are **not** in version control, so a fresh
checkout has none.

:func:`grand.geo.topography.update_data` downloads what a region needs.  A
lookup with no tile returns ``nan`` rather than raising — see
:doc:`troubleshooting`.

Geomagnetic field
-----------------

``data/geomagnet/`` holds the two coefficient files, and both **are** in version
control, so :mod:`grand.geo.geomagnet` works on a fresh checkout.

``IGRF13.COF`` is IGRF-13, whose published validity ended on 1 January 2025.
It still evaluates outside that window, silently.  IGRF-14 was released in
2024 and has not been adopted here; see :doc:`known_issues`.
