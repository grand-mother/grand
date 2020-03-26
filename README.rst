grand package |workflow| |codecov| |docs| |appimage|
====================================================

*Core* `Python 3`_  *package for the offline software of the* `GRAND`_
*Collaboration.*


Quick start
-----------

The grand package can be installed from `PyPI`_ using pip, e.g. as:

.. code:: bash

   python3.8 -m pip install -U --user grand

Note that Python 3.8 or above is required. On Linux you can get it as an
`AppImage`_ including grand dependencies as:

.. code:: bash

   wget https://github.com/grand-mother/grand/releases/download/appimage/python3-x86_64.AppImage
   chmod u+x python3-x86_64.AppImage

Check the `online documentation`_ for further details.


How to contribute
-----------------

Issues can be `reported on GitHub`_.

You can also contribute back to the code with Pull Requests `PR`_. Note that you
first need to fork and clone this repository. On Linux a development
environment is provided. It can be enabled by sourcing the `setup.sh`_ script.

In order to be accepted your changes are expected to successfully pass the
integration tests. You can run does locally as:

.. code:: bash

   python -m tests


License
-------

The GRAND software is distributed under the LGPL-3.0 license. See the provided
`LICENSE`_ and `COPYING.LESSER`_ files.


.. Local links

.. _COPYING.LESSER: https://github.com/grand-mother/grand/blob/master/COPYING.LESSER

.. _LICENSE: https://github.com/grand-mother/grand/blob/master/LICENSE

.. _setup.sh: https://github.com/grand-mother/grand/blob/master/env/setup.sh


.. Externals links

.. _AppImage: https://github.com/grand-mother/python/releases/download/continuous/python3-x86_64.AppImage

.. _GRAND: http://grand.cnrs.fr

.. _online documentation: https://grand-mother.github.io/grand-docs

.. _PR: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests

.. _PyPI: https://pypi.org/project/grand

.. _Python 3: https://www.python.org

.. _reported on GitHub: https://github.com/grand-mother/grand/issues


.. Badges

.. |appimage| image:: https://img.shields.io/badge/python3-x86_64-blue.svg
   :target: `AppImage`_

.. |codecov| image:: https://codecov.io/gh/grand-mother/grand/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/grand-mother/grand

.. |docs| image:: https://img.shields.io/badge/docs-ready-brightgreen.svg
   :target: `online documentation`_

.. |workflow| image:: https://github.com/grand-mother/grand/workflows/Tests/badge.svg
   :target: https://github.com/grand-mother/grand/actions?query=workflow%3ATests
