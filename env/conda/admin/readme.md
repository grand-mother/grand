
# First install GRANDLIB

```
conda create -c conda-forge --name grandlib-root root scipy numba python=3.9
```

## Install grandlib with pip

```
conda activate grandlib-root

pip install -r env/conda/requirements.txt
pip install -r quality/requirements.txt
```

## Add direction include from conda env for gull/turtle compilation

```
CONDA_PREFIX=/home/jcolley/miniconda3/envs/grandlib-root
export C_INCLUDE_PATH=/home/jcolley/miniconda3/envs/grandlib-root/include
export LIBRARY_PATH=/home/<?????>/miniconda3/envs/grandlib/lib
```
