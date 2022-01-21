# GRAND environnement configuration

## Libraries compilation in place under conda 

```
# Checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout dev_noAppim

# Create env conda "grand_user" (only one time)
conda env create -f env/conda/grand_user.yml

# Init env
conda activate grand_user
source env/setup.bash

# Tests
which python
pytest tests
cd examples/simulation/
python shower-event.py

# Leave grand env
conda activate
```



## With AppImage
```
# Checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout dev

# Init env
source env/setup_AppImage.bash

# Tests
which python
pytest tests
cd examples/simulation/
python shower-event.py
```