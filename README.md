# GRANDlib

[![tests](https://github.com/grand-mother/grand/actions/workflows/tests.yml/badge.svg)](https://github.com/grand-mother/grand/actions/workflows/tests.yml)
[![Code Quality](https://github.com/grand-mother/grand/actions/workflows/lint.yml/badge.svg)](https://github.com/grand-mother/grand/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/grand-mother/grand/branch/main/graph/badge.svg)](https://codecov.io/gh/grand-mother/grand)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://grand-mother.github.io/grand-docs)
[![PyPI](https://img.shields.io/pypi/v/grand.svg)](https://pypi.org/project/grand/)
[![arXiv](https://img.shields.io/badge/arXiv-2408.10926-orange.svg)](https://arxiv.org/abs/2408.10926)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2024.109461-blue.svg)](https://doi.org/10.1016/j.cpc.2024.109461)
[![INSPIRE](https://img.shields.io/badge/INSPIRE-cited%20by-003a6c.svg)](https://inspirehep.net/literature?q=refersto%20recid%202821264)
[![License: LGPL-3.0](https://img.shields.io/badge/License-LGPL--3.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Offline data handling, simulation and analysis for the
[Giant Radio Array for Neutrino Detection](http://grand.cnrs.fr).

> **Some badges above are not green yet.** Continuous integration is being
> rebuilt and the package is not yet on PyPI. They are here so that they turn
> green as each piece lands rather than being added after the fact. See
> [`docs/dev/RECOVERY_PLAN.md`](docs/dev/RECOVERY_PLAN.md).

## What it does

GRANDlib performs end-to-end simulation of the detector: from an electric
field produced by an external air-shower simulation, through the antenna
response, the Galactic noise and the radio-frequency chain, to the digitized
voltage a detection unit records. It also defines the data format the
collaboration uses, and the coordinate systems everything is expressed in.

It deliberately does little physics itself. Air showers and their radio
emission come from ZHAireS or CoREAS, tau propagation from DANTON, terrain
from TURTLE, the geomagnetic field from GULL. What GRANDlib owns is the
schema, the frame conversions, and the instrument response.

## Installation

```bash
conda env create -f env/conda/grand-dev.yml --solver=libmamba
conda activate grand-dev
source env/setup.sh
```

`env/setup.sh` compiles the TURTLE and GULL bindings and downloads the data
model. Full instructions, including what the environment provides and what
does not work yet, are on the [installation
page](docs/source/installation.rst).

## Quickstart

```python
from grand import Efield2Voltage

signal = Efield2Voltage("input_efield.root", "output_voltage.root")
signal.params["add_noise"]    = True
signal.params["add_rf_chain"] = True
signal.compute_voltage()
```

or from a shell:

```bash
python scripts/convert_efield2voltage.py input_efield.root -o output_voltage.root
```

## Documentation

Build it locally:

```bash
cd docs && make html
```

The narrative pages cover [coordinate
systems](docs/source/coordinates.rst) — the largest source of user error —
the [data model](docs/source/datamodel.rst), the [code
architecture](docs/source/architecture.rst), and [known
issues](docs/source/known_issues.rst).

## Contributing

Issues can be [reported on
GitHub](https://github.com/grand-mother/grand/issues). Pull requests are
welcome; fork and clone first.

## Citing

If GRANDlib contributes to work you publish, please cite:

> R. Alves Batista *et al.* (GRAND Collaboration), *GRANDlib: A simulation
> pipeline for the Giant Radio Array for Neutrino Detection (GRAND)*,
> Comput. Phys. Commun. **308** (2025) 109461,
> [arXiv:2408.10926](https://arxiv.org/abs/2408.10926).

```bibtex
@article{GRAND:2024atu,
    author        = "Alves Batista, Rafael and others",
    collaboration = "GRAND",
    title         = "{GRANDlib: A simulation pipeline for the Giant Radio Array for Neutrino Detection (GRAND)}",
    eprint        = "2408.10926",
    archivePrefix = "arXiv",
    primaryClass  = "astro-ph.IM",
    doi           = "10.1016/j.cpc.2024.109461",
    journal       = "Comput. Phys. Commun.",
    volume        = "308",
    pages         = "109461",
    year          = "2025"
}
```

## License

LGPL-3.0. See [LICENSE](LICENSE) and [COPYING.LESSER](COPYING.LESSER).
