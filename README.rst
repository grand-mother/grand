grand package |workflow| |codecov| |docs| |appimage|
====================================================

*Core* `Python 3`_  *package for the offline software of the* `GRAND`_
*Collaboration.*


Environment
-----------

GRAND library can be used under `docker`_ to define a correct environment, else you must install `ROOT`_ library and compile `TURTLE`_ and `GULL`_ library under your computer.

Don't forget to initialize grand library before use it with script **env/setup.sh** only in the root of the package

.. code:: bash
   
   $ pwd
   .../grand
   $ source env/setup.sh


Documentation
----------- 

Check the `online documentation`_ for further details.


How to contribute
-----------------

Issues can be `reported on GitHub`_.

You can also contribute back to the code with Pull Requests `PR`_. Note that you
first need to fork and clone this repository. On Linux a development
environment is provided. See env/readme.md, a docker environnement is provided.



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

.. _docker: https://github.com/grand-mother/grand/wiki

.. _ROOT: https://root.cern/install/

.. _TURTLE: https://github.com/niess/turtle

.. _GULL: https://github.com/niess/gull

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
