# Installation

## With conda

```
# git checkout
git clone https://github.com/grand-mother/grand.git
cd grand/
git checkout dev_noAppim

# create env conda "grand_user" (only one time)
conda env create -f env/conda/grand_user.yml

# init env
conda activate grand_user
source env/setup.bash

# Tests
which python
pytest tests
cd examples/simulation/
python shower-event.py
```

