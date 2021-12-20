# Code quality tools for developper

Initialize code quality tools in current env.

After 

```console
source env/setup.sh
```
do 

```console
python -m pip install -r quality/requirements.txt
```

## Check code, static analysis

We use [pylint](https://www.pylint.org/) as static code analysis to check coding standard (PEP8) and as error detector.

### Configuration 

With file 

```console
quality/pylint.conf
```
main options:

- 'disable' a rule
- 'enable' a rule
- 'ignored-classes' for false positive, for example astropy.units

### Script for grand

```console
grand_quality_analysis.bash
```

### Output example

```console
grand/tools/fake.py:20: [E0602(undefined-variable), max_2_vectors] Undefined variable 'vb'
grand/tools/fake.py:45: [E1111(assignment-from-no-return), min_2_vectors_pos] Assigning result of a function call, where the function has no return
grand/tools/fake.py:47: [E1137(unsupported-assignment-operation), min_2_vectors_pos] 'v_min' does not support item assignment
```

You can see the module path and line number where pylint detected a problem and then a description of the problem

## Check test coverage : 

We use [coverage.py](https://coverage.readthedocs.io/en/stable/) for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.

### Configuration 

We use this simple option 

```console
coverage run --source=grand -m pytest tests 
```

### Script for grand

```console
grand_quality_test_cov.bash
```

### Output example

```console
Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
grand/__init__.py                               6      0      0      0   100%
grand/io.py                                   307     23    116     20    89%
grand/libs/__init__.py                          6      0      0      0   100%
grand/libs/gull.py                             71      4     16      3    92%
grand/libs/turtle.py                          170     30     64     20    78%
grand/logging.py                               67     25      8      0    56%
grand/simulation/__init__.py                    4      0      0      0   100%
grand/simulation/antenna/__init__.py            3      0      0      0   100%
grand/simulation/antenna/generic.py           104     53     26      2    45%
grand/simulation/antenna/tabulated.py         125     86     18      0    30%
grand/simulation/pdg.py                        25      0      0      0   100%
grand/simulation/shower/__init__.py             4      0      0      0   100%
grand/simulation/shower/coreas.py             148     10     38      8    90%
grand/simulation/shower/generic.py            136     10     40      4    91%
grand/simulation/shower/zhaires.py            161     40     32      7    73%
grand/store/__init__.py                         2      0      0      0   100%
grand/store/protocol.py                        20      2      0      0    90%
grand/tools/__init__.py                         6      0      0      0   100%
grand/tools/coordinates/__init__.py             4      0      0      0   100%
grand/tools/coordinates/frame.py              206      4     64      8    96%
grand/tools/coordinates/representation.py      50      1     10      1    97%
grand/tools/coordinates/transform.py           76      0     22      1    99%
grand/tools/geomagnet.py                       53      0     16      0   100%
grand/tools/topography.py                     169     12     50      6    91%
-----------------------------------------------------------------------------
TOTAL                                        1923    300    520     80    82%
```

coverage.py provides also pretty HTML ouput page by module that indicate zone coverage. 
Open file 

```console
quality/html_coverage/index.html
```
with a web navigator.

### Launch only one test

During a debugging phase, it is practical to launch only the test you are working on, this is possible with the following command

```console
pytest tests/tools/test_fake.py::test_to_fix
```

At the end of development re-launched

```console
grand_quality_test_cov.bash
```
to update coverage with correction.


## Check type/annotation

We use [mypy]. Mypy is an optional static type checker for Python that aims to combine the benefits of dynamic (or "duck") typing and static typing

### Configuration 

We use this simple option 

```console
--config-file=tests/mypy.ini 
```

### Script for grand

```console
grand_quality_type.bash
```

### Output example


See report in file 

```console
quality/report_type.txt
```

or HTML page here

```console
quality/html_mypy/index.html
```

with web navigator.


# SonarQube server for grand

SonarQube server for grand project is available [here](https://sonarqube.grand.in2p3.fr).

All users have a personal project associated to last branch pushed on github to improve the code quality.

Ask to Colley JM login and password.


