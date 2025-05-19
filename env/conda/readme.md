# GRANDLIB env with conda

## Frequent confusion

The conda environment **does not contain GRANDLIB code**. The conda environment installs the necessary libraries, executables, initializes environment variables to correctly compile the C part of GRANLIB (gull and turtle), please read [constraints to use grandlib](https://github.com/grand-mother/grand/wiki#constraints-to-use-grandlib) in GRANDLIB wiki.

## WARNING

 * A wifi connection is not recommended for this installation
 * GRANDLIB conda environnement is only available with **amd64 architecture**, for arm64 work in progress. 

## miniconda installation

You need to have conda command installed, the minimal package to have it is miniconda. The [page](https://www.anaconda.com/docs/getting-started/miniconda/install) to install miniconda

## conda doc

[CLI conda environnement](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

[Conda cheat sheet PDF](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

## Some conda commands

* udpdate conda version

```
conda update conda
```

* list of available user environments

```
conda env list
```

* open/activate user environments

```
conda activate <my_env>
```

* close/deactivate user environments

```
conda deactivate
```

## Import GRANDLIB environnement for amd64 processor architecture

Start by update your conda and create the GRANDLIB environnement with file conf defined in grand/env/conda 

```
conda config --set channel_priority disabled
conda env create -n grandlib --file grandlib_amd64.yml
conda activate grandlib
```



Initialize GRANDLIB package and GULL/TURTLE compilation, in 
the root package, first clone the package grand

```
git clone https://github.com/grand-mother/grand.git
cd grand
source env/setup.sh
```
### Compilation failed in other environment

If you have already tried to compile the package in an incorrect environment you must clean the compilation files already produced to start from scratch with `make clean` in `grand/src` directory

```bash
cd src
make clean
cd ..
source env/setup.sh
```

Finally launch tests suite package to

```
python -m pip install -r quality/requirements.txt
grand_quality_test_cov.bash
```


## Creating a grandlib conda environement from scratch
You can create a conda environement for grandlib using the following recipie :

### Create a conda environnement for ROOT version 6.30.4 :
```
conda create -c conda-forge --name grandlib_root_6.30 root==6.30.4
```
### Activate your new environnement
```
conda activate grandlib_root_6.30
```
### Export your new environnement into a yaml file
```
conda env export >grandlib_root_6.30.yml
```
### Edit your yaml file to add the requested libs
Add the following lines to the file (check /grand/env/conda/grandlib_amd64.yml for position) :
```
- pip:
  - appjar
  - asdf
  - asdf-standard
  - asdf-transform-schemas
  - asdf-unit-schemas
  - asteval
  - astroid
  - astropy
  - astropy-iers-data
  - awkward
  - awkward-cpp
  - bcrypt
  - black
  - click
  - contourpy
  - coverage
  - cramjam
  - cryptography
  - cycler
  - dill
  - fonttools
  - fsspec
  - future
  - greenlet
  - h5py
  - iniconfig
  - isort
  - jmespath
  - kiwisolver
  - lazy-object-proxy
  - lmfit
  - lxml
  - matplotlib
  - mccabe
  - mypy
  - mypy-extensions
  - pandas
  - paramiko
  - pathspec
  - pillow
  - plotly
  - pluggy
  - psycopg2-binary
  - py
  - pyerfa
  - pylint
  - pynacl
  - pyparsing
  - pytest
  - scipy
  - scp
  - semantic-version
  - sqlalchemy
  - sshtunnel
  - tenacity
  - tokenize-rt
  - toml
  - tomlkit
  - tzdata
  - uncertainties
  - uproot
  - wrapt
```
### Deactivate your conda env and recreate one from the new file

```
conda deactivate
conda env create -n grandlib_root_6.30_complete --file grandlib_root_6.30.yml
conda activate grandlib_root_6.30_complete
```


If you encounter a problem write a ticket [here](https://github.com/grand-mother/collaboration-issues/issues)
