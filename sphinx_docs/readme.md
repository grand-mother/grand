This Sphinx Documentation can be viewed in any browser or VS-Code html viewer extension by opening the _build/html/index.html file.
Sphinx requires doctring style commands.

**Requirements**
- `pip install sphinx`
- `pip install sphinx-rtd-theme`

**Updating the Documentation**
- run `make html` in this folder to update current modules
- add module name in the conf.py file if new modules are added
- run `sphinx-apidoc -o sphinx_docs .` in the current supfolder (where the modules are saved)