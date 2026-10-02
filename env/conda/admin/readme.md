
# First install GRANDLIB

```
conda create -c conda-forge --name grandlib-root root scipy numba python=3.12
```

## Install grandlib with pip

```
conda activate grandlib-root

pip install -r env/conda/requirements.txt
pip install -r quality/requirements.txt
```

## Add direction include from conda env for gull/turtle compilation

```
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/lib
```

## export conda env

```
  conda env export > my_env.yml
```