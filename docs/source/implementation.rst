Implementation notes
====================

.. contents::
   :local:

Details that are not obvious from reading the code, and that have cost
someone time at least once.

Coordinate objects are NumPy arrays
-----------------------------------

Every frame — :class:`~grand.geo.coordinates.Geodetic`,
:class:`~grand.geo.coordinates.ECEF`, :class:`~grand.geo.coordinates.LTP`,
:class:`~grand.geo.coordinates.GRANDCS` — subclasses :class:`numpy.ndarray`.
That buys arithmetic and broadcasting for free, and costs two things.

Instance attributes do not survive a plain copy: ``ndarray`` subclasses do not
carry their ``__dict__`` through :func:`copy.copy`, so the origin, basis and
reference level would be lost.  :func:`grand.geo.coordinates.copy` copies the
attributes explicitly, and is what should be used.

And a component is an array, not a scalar.  ``float(position.x)`` works only
while the object holds a single point; under NumPy 2 it raises on anything
else.  Use ``np.ravel(...)[0]`` where a scalar is genuinely wanted.

The tree classes are descriptor-driven dataclasses
---------------------------------------------------

A field on a tree class is not an ordinary attribute.  ``StdVectorListDesc``,
``TTreeScalarDesc`` and their siblings are descriptors that translate between
Python values and the C++ types a ``TTree`` branch holds, so declaring

.. code-block:: python

    nutrig_rhox: StdVectorListDesc = field(default=StdVectorListDesc("unsigned short"))

adds a branch to the on-disk format.  That is why a field addition is a change
to a data contract rather than an implementation detail, and why
``tests/dataio/test_schema_snapshot.py`` exists.

It also has a sharp edge worth knowing.  When a dataclass field takes its
default, the descriptor object itself is passed to ``__set__``; the default
has already been installed by ``create_default``, so there is nothing to
assign.  Getting that wrong made every tree class impossible to construct
under NumPy 2 — see the changelog.

Importing the package requires ROOT
------------------------------------

``grand/dataio/descriptors.py`` evaluates ``ROOT.gROOT.GetVersionInt() >= 63600``
at **module import time**, and :mod:`grand.geo.topography` imports
:mod:`grand.dataio.protocol`.  Between them, importing anything under
``grand`` pulls in the full ROOT runtime — including importing it in order to
document it, which is why ``docs/source/conf.py`` carries a typed mock for
builds outside the environment.

The dependency also runs the wrong way round: geometry should not need the
data layer.  See :ref:`issue-import-requires-root`.

Angles are in degrees
---------------------

Throughout :mod:`grand.geo.coordinates`, and in the antenna response.  The
spherical polar angle is measured down from the zenith while horizontal
elevation is measured up from the horizon, and azimuth is measured from north
while spherical :math:`\phi` is measured from the :math:`+x` axis — so the two
run in opposite senses.  :doc:`coordinates` shows this executing.

Frequencies are in MHz, times in ns
------------------------------------

By convention encoded in parameter names: ``freqs_mhz``, ``f_samp_mhz``,
``dt_ns``.  Where a name does not carry the unit, the docstring states it.
The Galactic-noise tables are voltage spectral densities in µV/MHz, converted
to a per-bin amplitude by multiplying by :math:`\sqrt{\Delta f}`.

.. warning::

   **The antenna model is the exception.**
   :class:`~grand.sim.detector.antenna_model.AntennaModel` stores its frequency
   axis in **hertz**, not megahertz:

   .. code-block:: python

       >>> AntennaModel().leff_sn.frequency[[0, -1]]
       array([3.0e+07, 2.5e+08])

   Everything in :mod:`grand.sim.detector.rf_chain` takes and returns MHz, so
   code that reads a frequency from the antenna model and hands it to the RF
   chain is wrong by a factor of :math:`10^6`.  The attribute name carries no
   unit suffix, which is what makes this easy to miss.

   Notebook 03 divides by ``1e6`` at every use for this reason.

The RF chain cascades in ABCD form
-----------------------------------

Scattering parameters are what a vector network analyser measures, but they do
not cascade: the S-matrix of two networks in series is not the product of
their S-matrices.  :func:`~grand.sim.detector.rf_chain.s2abcd` converts to the
transmission representation, which does, and that is the only reason the
conversion exists.  A matched, lossless through-line gives the identity
matrix — a cheap invariant for any test of that function.

FFT lengths are rounded up
--------------------------

:func:`~grand.sim.efield2voltage.get_fastest_size_fft` rounds the transform
length up to the next size with a small prime factorisation, and returns the
matching frequency axis.  It reads only ``f_samp_mhz[0]``, so an event whose
detection units sampled at different rates would silently get the first unit's
frequency axis applied to all of them.

Galactic noise is generated per unit, independently
----------------------------------------------------

Spatial coherence between neighbouring detection units is not modelled — it is
expected to be small given how sparse the array is — so each unit receives its
own realisation of the sky-averaged noise.  Note that the published
description and the implementation differ in *how* that realisation is drawn;
see :ref:`issue-galactic-noise-normalisation`.
