grand package |travis| |codecov| |docs| |appimage|
==================================================

*This is the core package for the GRAND software. It is powered by* `Python 3`_,
`Scientific Python`_, `Astropy`_ *and vitamins. You can download does bundled in
a single* `AppImage`_. *Except vitamins.*


Quickstart
----------

Clone the present repository and source the `setup.sh`_ file:

.. code:: bash

   git clone https://github.com/grand-mother/grand
   cd grand
   source env/setup.sh

Check the `online documentation`_ for further details.




How to contribute to the code development
----------

* If you are new to github, please, create a user account and send your username to Olivier or Valentin#1 to get added to the grand-mother's poeple.
    - a useful start is to checkout provided Git Cheat Sheets, e.g. [link to one short git guide](https://www.digitalocean.com/community/tutorials/how-to-use-git-a-reference-guide) or [link to a sheet](https://education.github.com/git-cheat-sheet-education.pdf)

* clone this repository: ```git clone https://github.com/grand-mother/radio-simus.git```
* use ```git pull``` to get your repository up-to-date
* Implement your ideas/changes etc. Be aware that those ones are not in conflict with other user's needs. Therefore run the tests: ```python3 -m tests```
* Write up a documentation to your implemented function etc. (tool to be identified, worst case: docstring in code). Implement tests to prevent other user to break your needed functionality.
* Commit with messages and push: ```git commit -m '<what was done>' <file>; git push```.
    NOTE: This will be changed to a pull request soon-ish.
* Validate that travis build tests pass after commiting



How to use the code
----------

* clone the repository: ```git clone https://github.com/grand-mother/grand.git```
* use ```git pull``` to keep your repository up-to-date
* Work in your local copy of the repository. Implement your ideas/changes etc. 
* Report on issues on the code and give feedback on the usage.
* One can find several scripts for example applications in the folder 'examples'. To run them download Coreas example files (CoREAS_testEvents.zip) from the grand store [https://github.com/grand-mother/store/releases] 



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

.. _Python 3: https://www.python.org

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
