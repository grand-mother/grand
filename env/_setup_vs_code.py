#! /usr/bin/env python3


import sys, os

GRAND_ROOT = os.environ["GRAND_ROOT"]

env_vs_code = GRAND_ROOT+"/.env"

s_env_vs = f"""# file env for VS Code 
GRAND_ROOT={GRAND_ROOT}
PYTHONPATH={GRAND_ROOT}:$PYTHONPATH
PATH=./quality:./examples/dataio:./scripts:$PATH
"""

if os.path.exists(env_vs_code):
    print("PASSED : VS Code .env file already exits.To re-create a default one, remove it and restart 'source env/setup.sh'")
    sys.exit(0)

print('Create VS Code file environment.')

with open(env_vs_code,"w") as fenv:
    fenv.write(s_env_vs)
    
    