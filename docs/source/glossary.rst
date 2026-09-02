Glossary
========

GRAND carries jargon from three directions at once — radio astronomy, air-shower
physics and RF engineering — and a term that is obvious to one of those
communities is often opaque to the other two.  This page is the place to look
when a field name in :doc:`datamodel` or an argument in the :doc:`api` does not
explain itself.

.. glossary::
   :sorted:

   ADC
      Analogue-to-digital converter.  In GRAND, 14 bits over a 1.8 V full
      scale, giving a quantisation step of about 110 µV.  Also the name of the
      tree that stores what a unit actually recorded, :class:`~grand.dataio.event_trees.TADC`.

   analysis level
      How far a file has been processed, written into the file name as ``_L0_``
      or ``_L1_`` and into the tree as ``analysis_level``.  The two must agree;
      see :doc:`datamodel`.

   balun
      *Balanced-to-unbalanced* transformer.  Converts between the antenna's
      balanced feed and the unbalanced coaxial line.  Two appear in the RF
      chain, one after the low-noise amplifier and one before the ADC.

   DC2
      The second GRAND Data Challenge: a collaboration-wide exercise in which
      a common simulated dataset is produced and analysed.  Much of the
      branch activity in the repository traces to it.

   DU
      Detection unit.  One antenna with its three arms, its electronics and its
      digitiser.  ``du_id`` identifies it; ``du_xyz`` is its position in the
      site frame.

   ECEF
      Earth-Centred, Earth-Fixed.  A Cartesian frame with its origin at the
      centre of the Earth, rotating with it.  GRANDlib uses it as the pivot
      through which all other frame conversions pass; see :doc:`coordinates`.

   effective length
      The vector :math:`\boldsymbol{\ell}(\nu, \theta, \phi)` that gives an
      antenna arm's response to an incoming field:
      :math:`V_{\mathrm{oc}} = \boldsymbol{\ell}\cdot\boldsymbol{E}`.  Complex,
      so the antenna disperses as well as scales.  Notebook 03.

   ellipsoidal height
      Height above the WGS-84 reference ellipsoid, which is what GPS reports
      and what :class:`~grand.geo.coordinates.Geodetic` stores.  Not the same
      as height above sea level; see :term:`geoid undulation`.

   geoid undulation
      The difference between the :term:`ellipsoidal height` and height above
      mean sea level, reaching ±100 m worldwide and −7.7 m at the
      GRANDProto300 site.  Notebook 07.

   GP13
      A thirteen-unit GRAND prototype array.  Appears as a ``du_type`` and in
      several file names.

   GP300
      GRANDProto300: the 300-unit prototype array at Dunhuang, China, and the
      default ``du_type`` throughout the simulation.

   GRANDCS
      The GRAND coordinate system: a local Cartesian frame with ``x`` pointing
      **north**, ``y`` west and ``z`` up.  Distinct from
      :class:`~grand.geo.coordinates.LTP` with ``orientation='ENU'``, where
      ``x`` is east.  Getting these two confused rotates a layout by 90°.

   GULL
      The geomagnetic-field library GRANDlib compiles from source, wrapping
      IGRF and WMM.  Built by ``src/Makefile``.

   Handle3dTraces
      The in-memory container for a set of three-component traces, in
      :mod:`grand.basis.traces_event`.  What most analysis code actually
      manipulates.

   HorizonAntenna
      The GRAND antenna design: three arms, two horizontal and one vertical,
      on a mast a few metres above the ground.

   LFMap
      A low-frequency sky map of Galactic brightness temperature.  Folded
      through the antenna response, it is the source of the Galactic-noise
      tables; notebook 05.

   LNA
      Low-noise amplifier.  The first active stage of the RF chain, and the one
      that sets the receiver noise figure.

   LST
      Local sidereal time, in hours.  The Galactic-noise level is tabulated
      against it, because the Galactic plane rises and sets on a sidereal day.

   LTP
      Local Tangent Plane.  A Cartesian frame tangent to the ellipsoid at a
      chosen origin, with a selectable ``orientation`` such as ``'ENU'`` or
      ``'NWU'``.

   NUTRIG
      The GRAND online trigger project.  Several tree fields carry
      correlation quantities produced by it; their naming is
      :ref:`an open question <issue-nutrig-field-names>`.

   open-circuit voltage
      :math:`V_{\mathrm{oc}}`, the voltage at the antenna terminals before any
      electronics.  The output of the antenna stage and the input to the RF
      chain.

   RF chain
      The cascade of two-port networks between the antenna and the ADC:
      matching network, LNA, baluns, cable, and a variable-gain amplifier with
      a filter.  Notebook 04.

   S-parameters
      Scattering parameters, what a vector network analyser measures.  They do
      not cascade by multiplication, which is why GRANDlib converts each stage
      to an ABCD matrix first.

   sim2root
      The tools that convert ZHAireS or CoREAS output into the GRAND schema.
      They live in ``sim2root/`` and are outside the lint and test gates; see
      :doc:`ci`.

   SRTM
      Shuttle Radar Topography Mission.  The elevation dataset TURTLE reads,
      shipped as one ``.hgt`` file per one-degree square.  Not in version
      control.

   TTree
      A ROOT data structure: a table whose columns (*branches*) can hold
      variable-length arrays.  Everything GRAND records or simulates lives in
      one.

   TURTLE
      The topography library GRANDlib compiles from source, which reads
      :term:`SRTM` tiles and performs ray–ground intersection.

   VGA
      Variable-gain amplifier.  Its setting is exposed as ``vga_gain`` and is
      currently :ref:`ignored <issue-vga-gain-ignored>`.

   Xmax
      The atmospheric depth at which an air shower reaches its maximum number
      of particles, in g/cm².  ``xmax_pos_shc`` is the corresponding position,
      expressed in the shower-core frame.

   ZHAireS
      One of the two air-shower simulation codes GRAND uses upstream; CoREAS is
      the other.  Neither is part of GRANDlib.
