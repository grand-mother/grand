# GRAND environnement configuration


## With docker (tested and used for github CI)

OS compatible: **linux and MacOS**

A docker image of GRAND environnement with root is available on [dockerhub](https://hub.docker.com/r/jcolley/grandlib_dev/tags).

```
# Checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout <mybranch>
cd ..

# create a docker container
docker pull jcolley/grandlib_dev:0.1
docker run -it --name grand_dev -v `pwd`:/home jcolley/grandlib_dev:0.1

# init in docker
==> in docker with GRAND env
root@7fdf26599527:/home#
root@7fdf26599527:/home# cd grand
root@7fdf26599527:/home/grand# . env/setup.sh 
...

ctrl+d to quit docker
<==

# to return to the container created
docker start -ia grand_dev
```


## With conda for linux user (must be tested)

OS compatible: linux

gull and turtle libraries compilation is not yet guaranteed with conda.
It's possible to install root toobwith

```
conda install -c conda-forge root
```

but not yet tested with grand environnement

```
# Checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout dev_noAppim

# Create env conda "grand_user" (only one time)
conda env create -f env/conda/grand_user.yml

# Init GRAND env
conda activate grand_user

source env/setup.sh

# Tests
which python
pytest tests
cd examples/simulation/
python shower-event.py

# Leave GRAND env
conda deactivate
```

## With conda for MacOS user (TODO)

OS compatible: MacOS

...


## With AppImage file

OS compatible: **linux only**

No longer supported in the current version, see tag master_ref2021
