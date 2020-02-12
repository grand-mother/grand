grand package |travis| |codecov| |docs| |appimage|
==================================================

*This is the development environment for the `grand` package. It is powered
by* `Python 3`_, `Scientific Python`_ and `Astropy`_. *You can download does
bundled in a single* `AppImage`_.


Quick start
-----------

Clone the present repository and source the `setup.sh`_ file:

.. code:: bash

   git clone https://github.com/grand-mother/grand
   cd grand
   source env/setup.sh

Check the `online documentation`_ for further details.


How to contribute
-----------------

Issues can be `reported on GitHub`_.

You can also contribute back to the code with Pull Requests `PR`_. Note that you
first need to fork this repository. Note also that in order to be accepted your
changes are expected to successfully pass the integration tests. You can run
does locally as:

.. code:: bash

   python -m tests


License
-------

The GRAND software is distributed under the LGPL-3.0 license. See the provided
`LICENSE`_ and `COPYING.LESSER`_ files.


.. Local links

.. _COPYING.LESSER: COPYING.LESSER

.. _LICENSE: LICENSE

.. _setup.sh: env/setup.sh


.. Externals links

.. _AppImage: https://github.com/grand-mother/python/releases/download/continuous/python3-x86_64.AppImage

.. _Astropy: https://www.astropy.org

.. _online documentation: https://grand-mother.github.io/grand-docs

.. _PR: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests

.. _Python 3: https://www.python.org

.. _reported on GitHub: https://github.com/grand-mother/grand/issues

.. _Scientific Python: https://www.scipy.org


.. Badges

.. |appimage| image:: https://img.shields.io/badge/python3-x86_64-blue.svg
   :target: `AppImage`_

.. |codecov| image:: https://codecov.io/gh/grand-mother/grand/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/grand-mother/grand

.. |docs| image:: https://img.shields.io/badge/docs-ready-brightgreen.svg
   :target: `online documentation`_

.. |travis| image:: https://travis-ci.com/grand-mother/grand.svg?branch=master
  :target: https://travis-ci.com/grand-mother/grand
