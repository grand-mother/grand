Installation
============

.. warning::
   The :mod:`grand` package is meant to be used on a Linux system. All its
   dependencies are bundled within an AppImage_. For developping on OSX or
   Windows we recommend using a virtual machine, E.g.  VirtualBox_ with
   Debian_. Note that on Windows 10, WSL does not currently support AppImages.

The :mod:`grand` package sources are available from GitHub_. Installation is
as simple as cloning the repository:

.. code-block:: bash

   git clone https://github.com/grand-mother/grand


Setting up the environment
--------------------------

Before using the package you must source the provided `setup.sh` file, as:

.. code-block:: bash

   source grand/setup.sh

This will configure your environment for using GRAND's `python` and its related
executables -located under grand/bin-, E.g. `pip`, `sphinx-build`, etc.

.. note::
   The GRAND `python` is isolated from your system and home space. User specific
   data can be found under `grand/user/grand`.

You can restore your initial environment by sourcing the `clean.sh` file.


Installing custom packages
--------------------------

Extra Python packages can be (un)installed within the GRAND environment with
`pip`, E.g. as:

.. code-block:: bash

   pip install test-pip-install

.. note::
   The packages are installed to the GRAND user space, i.e. in
   grand/user/grand/.local. You can get rid of them by simply deleting the
   grand/user folder.


Relocating
----------

The installation can be relocated, E.g. with `cp` or `mv`. Only, one needs to
source the `setup.sh` file again, once relocated.

.. warning::
   Custom packages, installed with `pip`, might not support relocation, E.g. to
   another Linux system. You can re-install them in this case.


.. _AppImage: https://appimage.org
.. _Debian: https://www.debian.org
.. _GitHub: https://github.com/grand-mother/grand
.. _VirtualBox: https://www.virtualbox.org
