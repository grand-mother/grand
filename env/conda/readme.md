# GRANDLIBenv with conda

Don't used Wifi for this installation


## update your conda version

conda update conda


## install root with conda

conda create -c conda-forge --name grandlib-root root python=3.9

## install grandlib with pip

conda activate grandlib-root

pip install -r env/conda/requirements.txt
pip install -r quality/requirements.txt

## add direction include from conda env for gull/turtle compilation

export C_INCLUDE_PATH=/home/jcolley/miniconda3/envs/grandlib-root/include
