# Installation

## With conda

```
# git checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout dev_noAppim

# create env conda (only one time)
conda env create -f environment.yml

# init env
conda activate grand_user
source env/setup.bash

# Tests
pytest tests
cd examples/simulation/
python shower-event.py
```

