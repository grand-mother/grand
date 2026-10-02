# GRANDlib

[![tests](https://github.com/grand-mother/grand/actions/workflows/tests-conda.yml/badge.svg?branch=dev-next)](https://github.com/grand-mother/grand/actions/workflows/tests-conda.yml)
[![Code Quality](https://github.com/grand-mother/grand/actions/workflows/lint.yml/badge.svg?branch=dev-next)](https://github.com/grand-mother/grand/actions/workflows/lint.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://grand-mother.github.io/grand)
[![arXiv](https://img.shields.io/badge/arXiv-2408.10926-orange.svg)](https://arxiv.org/abs/2408.10926)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2024.109461-blue.svg)](https://doi.org/10.1016/j.cpc.2024.109461)
[![License: LGPL-3.0](https://img.shields.io/badge/License-LGPL--3.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

GRANDlib is the software library of the
[Giant Radio Array for Neutrino Detection](http://grand.cnrs.fr) (GRAND).
It takes the radio pulse that an air-shower code (ZHAireS or CoREAS) computes
at each antenna and turns it into the voltages and ADC counts that a GRAND
detection unit records: antenna response, Galactic noise, RF chain and
digitization.  It also defines the ROOT data format the collaboration stores
simulated and measured data in, the coordinate frames, terrain and
geomagnetic field that tie them to real sites and tools to reconstruct a
shower from recorded signals.

Development happens on the `dev-next` branch.

## Installation

GRANDlib runs on Linux x86-64 with Python 3.10 or later and needs ROOT.  The
conda environment in this repository provides everything:

```bash
git clone https://github.com/grand-mother/grand.git
cd grand
conda env create -f env/conda/grand-dev.yml --solver=libmamba
conda activate grand-dev
source env/setup.sh
```

`env/setup.sh` compiles the TURTLE and GULL libraries and downloads about 1 GB
of model data.  With ROOT already installed, `pip install -e ".[dev]"`
followed by `source env/setup.sh` works too; GRANDlib is not on PyPI yet.  The
[installation page](docs/source/installation.rst) covers both routes and
Docker.

## Quick start

Simulate the voltages of the shower that ships with the repository, from the
repository root:

```python
from grand import Efield2Voltage

sim = Efield2Voltage("sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000",
                     "voltage.root", output_directory=".", seed=1, efield_level=0)
sim.compute_voltage()        # 44 antennas, about 10 s on one core
```

or, from a shell, voltage then ADC counts:

```bash
python scripts/convert_efield2voltage.py <simulation folder> --lst 18
python scripts/convert_voltage2adc.py <simulation folder>
```

The [quick start guide](docs/source/quickstart.rst) continues from here.

## Documentation

The documentation covers installation, a quick start, recipes for common
tasks, the coordinate conventions and units, the data format, the simulation
chain and an API reference.  Build it locally with

```bash
cd docs && make html         # then open docs/build/html/index.html
```

Twelve worked [notebooks](notebooks/) show each part of the library with
figures and can be read on GitHub without running them.

## Reporting problems and contributing

Report problems in the [issue tracker](https://github.com/grand-mother/grand/issues);
the [known issues](docs/source/known_issues.rst) page lists the open ones that
affect results.  Pull requests are welcome; the
[contributing guide](docs/source/contributing.rst) describes the checks a
change must pass.

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

## Acknowledgments

The GRAND Collaboration acknowledges the support from the National Science
Centre Poland for NCN OPUS grant no. 2022/45/B/ST2/02889.

## License

LGPL-3.0-or-later.  See [LICENSE](LICENSE) and [COPYING.LESSER](COPYING.LESSER).
