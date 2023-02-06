# GRANDLIBenv with conda

Don't used Wifi for this installation

conda doc

[CLI conda environnement](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

[conda-cheatsheet](file:///home/jcolley/T%C3%A9l%C3%A9chargements/conda-cheatsheet-1.pdf)

## Some conda commands

```
conda env list

conda activate <my_env>

conda deactivate
```

## Update your conda version

```
conda update conda
```

## Import GRANDLIB environnement

Find file grandlib_env.yml in grand/env/conda directory

```
conda env create -n grandlib --file grandlib_env.yml
conda activate grandlib
```

Add path to include directory of grandlib conda env 

```
export C_INCLUDE_PATH=/home/<?????>/miniconda3/envs/grandlib/include
```

Initialize GRANDLIB package and GULL/TURTLE compilation, in 
the root package, do 

```
source env/setup.sh
```

Finally launch tests suite package to

```
grand_quality_test_cov.bash
```

## First install GRANDLIB

```
conda create -c conda-forge --name grandlib-root root python=3.9
```

### Install grandlib with pip

```
conda activate grandlib-root

pip install -r env/conda/requirements.txt
pip install -r quality/requirements.txt
```

### Add direction include from conda env for gull/turtle compilation

```
export C_INCLUDE_PATH=/home/jcolley/miniconda3/envs/grandlib-root/include
```