.. This page is generated from resources/GRANDlib_Handbook.zip
   by docs/dev/build_handbook.py.  Do not edit it by hand.

Installation
============


The GRAND library can be used under docker to define a correct environment, read `GRANDwiki <https://github.com/grand-mother/grand/wiki>`__ for more information, else you must install `ROOT <https://root.cern/install/>`__ library and compile `TURTLE <https://github.com/niess/turtle>`__ and `GULL <https://github.com/niess/gull>`__ library under your computer.
You can run the GRANDlib on you PC on a Docker container or on a Conda enviroment

Installation via Docker
=======================

In order to be able to run the GRANDlib on a Docker container you need to download `Docker <https://www.docker.com/get-started/>`__. After that you have to install the grandlib/dev:x image.

- **Tag 1.x:** for ``amd64`` architecture

  - 1.0: First version with ROOT 6.26/02, Python 3.8.10

  - 1.1: Updated Python packages, added DB support

  - 1.2: Fixed DB support, removed Doxygen

- **Tag 2.x:** for ``arm64`` architecture

  - 2.0: Fedora-based, initial release (Known issues: interactive plots,jupyter notebook)

Now we can start the installation process. Opening the terminal on the path you want to install/run GRANDlib and execute the following commands.

#. **Clone the repository:** ``git clone https://github.com/grand-mother/grand.git``

#. **Navigate to the directory:** ``cd grand/``

#. **Pull the Docker image:** ``docker pull grandlib/dev:x``

#. **Run Docker (ephemeral container):** ``docker run -it --rm -v $PWD:/home/grandlib/dev:x``

#. **Run Docker (named container):** ``docker run -p 8888:8888 -it --name container-name -v $PWD:/home/ grandlib/dev:x``

#. **Restart named container:** ``docker start -ia container-name``

#. **Initialize Grand Library:** ``cd grand`` then ``source env/setup.sh``

If the installation was done correctly you will get the following message:

.. code:: bash

   Set var GRAND_ROOT=/home
       ==============================
       add grand/quality to PATH
       ==============================
       add scripts to PATH 
       ==============================
       add grand to PYTHONPATH
       ==============================
       add AIRESBINDIR
       =============================
       Install external lib gull and turtle
       ====================================
       make: Nothing to be done for 'all'.
       ==============================
       Download data model (~ 452MB) for GRAND, please wait ...
       or
       Skip download data model
       

Installation via Conda
----------------------

**Note:**

**Option 1: Using the predefined Conda environment (recommended)**

#. Install Miniconda or Anaconda from: https://docs.conda.io/en/latest/miniconda.html

#. Update conda and configure channels:

   .. code:: bash

      conda update conda
      conda config --set channel_priority disabled 

#. Create and activate the environment:

   .. code:: bash

      git clone https://github.com/grand-mother/grand.git
        cd grand
        conda env create -f reqmt_grandenv_2509.yml
        conda activate grandenv_2509
        source env/setup.sh

#. If you compiled previously in a different environment, clean it:

   .. code:: bash

      cd src
      make clean
      cd ..
      source env/setup.sh

   *reqmt_grandenv_2509.yml* is present in the directory of this readme.md . In this requirement the library version are fixed with the aim that the entire GRAND community truly has an identical environment. This can cause an availability problem for certain distribution, in this case start to remove patch (last number), example

   matplotlib=3.10

   see if necessary minor (second number)

   matplotlib=3

#. (Optional) Run code quality and tests:

   .. code:: bash

      python -m pip install -r quality/requirements.txt
      grand_quality_test_cov.bash

**Option 2: Creating the environment manually (advanced)**

#. Create a base Conda environment with ROOT 6.30.4:

   .. code:: bash

      conda config --add channels defaults
      conda create  -c conda-forge  -n grandlib_root root=6.30.4
      conda activate grandlib_root
      pip install \
        appjar asdf asdf-standard asdf-transform-schemas asdf-unit-schemas \
        asteval astroid astropy astropy-iers-data awkward awkward-cpp bcrypt black \
        click contourpy coverage cramjam cryptography cycler dill fonttools fsspec \
        future greenlet h5py iniconfig isort jmespath kiwisolver lazy-object-proxy \
        lmfit lxml matplotlib mccabe mypy mypy-extensions pandas paramiko pathspec \
        pillow plotly pluggy psycopg2-binary py pyerfa pylint pynacl pyparsing \
        pytest scipy scp semantic-version sqlalchemy sshtunnel tenacity tokenize-rt \
        toml tomlkit tzdata uncertainties uproot wrapt

#. Export your environment:

   .. code:: bash

      conda env export > grandlib_root_6.30.yml 

#. Edit the YAML file and add the required Python packages (see Appendix).

#. Recreate the final environment:

   .. code:: bash

      conda deactivate
      conda env create -n grandlib_complete --file grandlib_root_6.30.yml
      conda activate grandlib_complete

If you encounter issues, report them at: https://github.com/grand-mother/collaboration-issues/issues

