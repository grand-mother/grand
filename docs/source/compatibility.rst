Platforms, versions and stability
=================================

Where GRANDlib runs, which versions are tested and what may change between
versions.

.. contents::
   :local:
   :depth: 1

Supported platforms
-------------------

=====================================  ======================================================
Platform                               Status
=====================================  ======================================================
Linux, x86-64                          Supported.  Every change is tested on Ubuntu.
macOS (Intel or Apple Silicon)         Not supported.  TURTLE and GULL, compiled at
                                       installation, have failed to build on Apple
                                       Silicon; see :doc:`installation` for a route
                                       reported to work.
Linux on ARM (aarch64)                 Not supported: not tested.
Windows                                Not supported; WSL is not tested.
Docker                                 Works (:doc:`installation`); no image is
                                       maintained.
=====================================  ======================================================

Tested versions
---------------

=========  ================================  =========================================
Component  Tested on every change            Notes
=========  ================================  =========================================
Python     3.12                              ``pyproject.toml`` accepts 3.10 and later;
                                             3.10 and 3.11 are not tested.
ROOT       6.36.04                           6.38.02 is also tested; a failure there
                                             does not block a change.
NumPy,     The versions in                   Installed by the conda
SciPy      ``env/conda/grand-dev.yml``       environment.
=========  ================================  =========================================

The conda environment in ``env/conda/grand-dev.yml`` pins these versions.
Installing it is the only route that matches what is tested.

Version numbers
---------------

Two numbers matter:

* **The package version**, ``0.1.0.dev1`` at present, printed by
  ``python -c "from grand import provenance; print(provenance.current())"``
  together with the git commit.  Snapshots are tagged ``v0.1.0-dev.N``.  The
  older tags ``v1.0.0`` (2022) and ``DC2*`` predate this numbering; do not
  read ``v1.0.0`` as newer.
* **The data-format version**, ``1.0.2``, in ``grand/dataio/version`` and
  tagged ``root_v1.0.2``.  It changes only when the ROOT trees change, on its
  own cycle.

Every voltage file records the package version and commit that wrote it
(``grandlib_version`` in ``TVoltage``, ``modification_software_version`` in
every tree).

What may change
---------------

GRANDlib is before its first release: under `Semantic Versioning
<https://semver.org/>`_, anything may change between ``0.x`` versions.  In
practice:

* **The data format** changes least.  A test compares every tree's layout
  with a stored snapshot, so any change is deliberate.  Files written by
  earlier versions are tested to still read.
* **The command-line options** in :doc:`commands` are tested against that
  page, so a change to one appears there.
* **The Python interface** of :mod:`grand.sim`, :mod:`grand.dataio` and
  :mod:`grand.geo` changes when a fix requires it.
* **The reconstruction**, :mod:`grand.analysis`, is recent and the most
  likely to change.

There is no deprecation period yet: a change that can break your code or
change your numbers is listed under *Changed* in the :doc:`changelog`; the
main ones are in :doc:`whatsnew`.  Read them before updating.

Reproducing a result
--------------------

Record the commit you ran; to reproduce, check it out again:

.. code-block:: bash

    git checkout v0.1.0-dev.27        # or the commit your files record
    source env/setup.sh

Updating to a newer version can change simulated numbers, as fixes have done
(:doc:`whatsnew`).

Where next
----------

* :doc:`installation`
* :doc:`whatsnew` and the :doc:`changelog`
