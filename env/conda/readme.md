# GRANDLIB env with conda

## WARNING

 * A wifi connection is not recommended for this installation
 * GRANDLIB conda environnement is only available with **amd64 architecture**, for arm64 work in progress. 

## miniconda installation

You need to have conda command installed, the minimal package to have it is miniconda. The [page](https://docs.conda.io/en/latest/miniconda.html) to install miniconda

## conda doc

[CLI conda environnement](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

conda cheatsheet PDF: file:///home/jcolley/T%C3%A9l%C3%A9chargements/conda-cheatsheet-1.pdf

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

Start by update your conda installation and create the GRANDLIB environnement with file conf defined in grand/env/conda 

```
conda env create -n grandlib --file grandlib_amd64.yml
conda activate grandlib
```



Initialize GRANDLIB package and GULL/TURTLE compilation, in 
the root package, do 

```
source env/setup.sh
```

Finally launch tests suite package to

```
python -m pip install -r quality/requirements.txt
grand_quality_test_cov.bash
```


If you encounter a problem write a ticket [here](https://github.com/grand-mother/collaboration-issues/issues)