Installation
============

.. warning::
   The :mod:`grand` package is meant to be used on Unix systems. For Windows
   users we recommend using WSL or a virtual machine.

Python 3.8 or later is required for installing and running the :mod:`grand`
package. OS specific instructions are found below.


Linux
-----

Once you have Python 3.8 installed your can get the :mod:`grand` package from
PyPI_, e.g. using pip as:

.. code-block:: bash

   pip install grand

If you don't already have Python3.8, on Linux we provide a ready to use
AppImage_. It contains Python3.8 and extra Python packages needed by
:mod:`grand`.  Note that you need not use this AppImage_. The :mod:`grand`
package is compatible with any distribution of CPython 3.8, e.g. installing from
source or using conda if you prefer.

If you like to go for the AppImage_ you can download it from GitHub_ as:

.. code-block:: bash
   wget https://github.com/grand-mother/grand/releases/download/appimage/python3-x86_64.AppImage
   chmod +x python3-x86_64.AppImage

Further instructions on `Python AppImages`_ can be found on GitHub. In
particular we recommend extracting the AppImage in order to get a contained and
expendable Python runtime. This can be done as:

.. code-block:: bash
   ./python3-x86_64.AppImage --appimage-extract
   mv squashfs-root python3
   rm -f python3-x86_64.AppImage

Then, you can export the AppImage_ python to you environment as:

.. code-block:: bash
   export PATH=$(path)/python3/usr/bin:$PATH


OSX
---

On OSX you will need to build the :mod:`grand` package from the source. This
can be done as:

.. code-block:: bash
   git clone https://github.com/grand-mother/grand.git
   cd grand
   pip3.8 install --user -U pip
   pip3.8 install --user -U -r requirements.txt
   make install PYTHON=$(which python3.8)

Once built, the :mod:`grand` package can be relocated to any desired location.


.. _AppImage: https://appimage.org
.. _GitHub: https://github.com/grand-mother/grand
.. _Python AppImages: https://github.com/niess/python-appimage
