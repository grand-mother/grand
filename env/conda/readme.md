# GRANDLIB env with conda

## Frequent confusion

The conda environment **does not contain GRANDLIB code**. The conda environment installs the necessary libraries, executables, initializes environment variables to correctly compile the C part of GRANLIB (gull and turtle), please [constraints to use grandlib](https://github.com/grand-mother/grand/wiki#constraints-to-use-grandlib) in GRANDLIB wiki.

## WARNING

 * A wifi connection is not recommended for this installation
 * GRANDLIB conda environnement is only available with **amd64 architecture**, for arm64 work in progress. 

## miniconda installation

You need to have conda command installed, the minimal package to have it is miniconda. The [page](https://docs.conda.io/en/latest/miniconda.html) to install miniconda

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


If you encounter a problem write a ticket [here](https://github.com/grand-mother/collaboration-issues/issues)
