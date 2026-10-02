GRANDlib: simulation and data handling for GRAND
================================================

GRANDlib is the software library of the `Giant Radio Array for Neutrino
Detection <http://grand.cnrs.fr>`_ (GRAND).  It takes the radio pulse that an
external air-shower code computes and turns it into the voltages and :term:`ADC`
counts that a GRAND :term:`detection unit <DU>` records.  The same package defines the
ROOT data format the collaboration stores its simulated and measured data in.
It also provides the coordinate frames, terrain and geomagnetic field that tie
both to real sites and a first set of reconstruction tools that go back from
recorded signals to the shower.

.. tip::

   **New here?**  Install it (:doc:`installation`), then simulate the voltages
   of the showers that ship with the repository:

   .. code-block:: python

       from grand import Efield2Voltage

       sim = Efield2Voltage("sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000",
                            "voltage.root", output_directory=".", seed=1, efield_level=0)
       sim.compute_voltage()        # two showers, about 10 s on one core

   Run it from the repository root.  ``efield_level=0`` reads the simulated
   electric field rather than its noisy level-1 copy.  The :doc:`quickstart`
   continues from here: reading the result back, switching stages off and
   digitizing.

.. important::

   **Important links**

   * :doc:`quickstart` and :doc:`installation`
   * :doc:`recipes`: short code for common tasks
   * :doc:`help`: questions and bug reports, on `GitHub
     <https://github.com/grand-mother/grand/issues>`_
   * :doc:`notebooks` (twelve worked notebooks)
   * :doc:`whatsnew`, :doc:`citing` and the :doc:`changelog`

**End to end.**  One call goes from the electric field at each antenna to the
voltage at the ADC input.  It projects the field on the antenna's effective
length, adds Galactic noise for the chosen sidereal time, passes the result
through the measured :term:`RF chain` and writes a file.  A second step digitizes it.
Each stage can also be called on its own arrays, without any file.

**One data format.**  Simulated and measured events are stored in the same
ROOT trees and read back as Python numbers and NumPy arrays.  Every voltage
file records the GRANDlib version that computed it.

**Tested against what it claims.**  Over 1200 tests check the package against
independent calculations, the shower codes' own output and properties that
must hold exactly (:doc:`testing`).  Most inputs are checked when a function
is called: a wrong unit or a missing input usually stops with a message that
names the argument, rather than producing a plausible wrong number.

Which pages do I need?
----------------------

**I analyze GRAND data.**  :doc:`installation`, the :doc:`cheatsheet`, then
:doc:`measured_data`, :doc:`datamodel` and the :doc:`data_format`,
:doc:`coordinates` and :doc:`sites`, the reading recipes in :doc:`recipes`
and notebooks 02, 09 and 11 (:doc:`notebooks`).

**I run simulations.**  :doc:`installation`, the :doc:`quickstart` and the
:doc:`tutorial`, then :doc:`simulation_production` from shower parameters to
ADC counts, :doc:`at_scale` for many showers, :doc:`commands`,
:doc:`simulation` for what each stage does and :doc:`known_issues` before
relying on absolute noise levels or the trigger.

**I change GRANDlib.**  :doc:`contributing`, :doc:`testing`, :doc:`ci` and
:doc:`architecture`.

What it can compute
-------------------

* The :term:`open-circuit voltage` at the three arms of a GRAND antenna, for any
  arrival direction, from tabulated HFSS, NEC or MATLAB effective lengths.
* Galactic noise from the :term:`LFMap` sky model at any :term:`local sidereal time <LST>`, for each
  detection unit independently.
* The response of the :term:`GRANDProto300 <GP300>` RF chain (matching network, LNA, baluns,
  cable, filter board) as a cascade of measured two-port networks.
* Digitization by the 14-bit, 500 MHz ADC, with saturation and optional
  measured noise and an offline version of the T1 trigger.
* Positions in geodetic, Earth-centered, local tangent-plane and array frames,
  ground elevation from :term:`SRTM` tiles, the geoid and the geomagnetic field.
* The arrival direction, the position of the shower maximum and an energy
  estimate, from the peak times and amplitudes of a recorded event
  (:mod:`grand.analysis`).

What it has been used for
-------------------------

* The GRANDlib paper (:cite:`GRAND:2024atu`): 300 :term:`ZHAireS` showers simulated
  across the full GRANDProto300 layout.
* The collaboration's Data Challenge datasets, converted with ``sim2root`` and
  processed with the voltage and ADC steps.
* Reading and viewing GRANDProto300 and :term:`GP13` data, including ten GP13
  cosmic-ray candidates that ship with the examples (notebooks 11 and 12).

When to use GRANDlib and when not
----------------------------------

GRANDlib is the tool for anything that needs the response of a GRAND
detection unit, or that reads and writes GRAND data.  It does not simulate
air showers or their radio emission (ZHAireS and CoREAS do).  It does not
model anthropogenic noise or spatial coherence of the Galactic noise between
units.  Its reconstruction is recent: the fits recover the showers their own
forward models generate, but have not yet been validated against full
simulations.

Citing
------

If GRANDlib contributes to work you publish, please cite
:cite:`GRAND:2024atu`.  :doc:`citing` gives the BibTeX entry.

GRANDlib is distributed under the GNU Lesser General Public License, version 3
or later.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installation
   quickstart
   tutorial
   cheatsheet
   recipes
   notebooks
   examples

.. toctree::
   :maxdepth: 2
   :caption: Analyzing data
   :hidden:

   measured_data
   datamodel
   coordinates
   sites
   writing_files

.. toctree::
   :maxdepth: 2
   :caption: Running simulations
   :hidden:

   simulation_production
   sim2root
   commands
   at_scale
   simulation
   validation

.. toctree::
   :maxdepth: 2
   :caption: When something goes wrong
   :hidden:

   troubleshooting
   known_issues
   logging
   help

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
   data_format
   data_files
   glossary
   handbook/index
   compatibility
   whatsnew
   changelog
   citing
   references

.. toctree::
   :maxdepth: 2
   :caption: For developers
   :hidden:

   contributing
   architecture
   testing
   ci
   roadmap
