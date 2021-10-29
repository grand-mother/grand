# Code quality tools for developper

Initialize code quality tools in current env.

After 

```console
. env/setup.sh
```
do 

```console
python -m pip install -r quality/requirement.txt
```

## Check code and syntax : pylint 

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
grand_quality_code.bash
```

### Output example

```console
Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention | x     |NC       |NC         |
+-----------+-------+---------+-----------+
|refactor   | x     |NC       |NC         |
+-----------+-------+---------+-----------+
|warning    | x     |NC       |NC         |
+-----------+-------+---------+-----------+
|error      | x     |NC       |NC         |
+-----------+-------+---------+-----------+

```

## Check test coverage : coverage.py


### Configuration 

TBD

### Script for grand

```console
grand_quality_cov_test.bash
```

### Output example

```console

========================================= test session starts ==========================================
platform linux -- Python 3.8.2, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: /home/jcolley/projet/grand_wk/test2/grand
collected 27 items                                                                                     

tests/core/test_io.py .                                                                          [  3%]
tests/libs/test_gull.py ..                                                                       [ 11%]
tests/libs/test_turtle.py ....                                                                   [ 25%]
tests/simulation/test_antenna.py ss                                                              [ 33%]
tests/simulation/test_shower.py ...                                                              [ 44%]
tests/store/test_protocol.py .                                                                   [ 48%]
tests/tools/test_coordinates.py ......                                                           [ 70%]
tests/tools/test_geomagnet.py ...                                                                [ 81%]
tests/tools/test_topography.py .....                                                             [100%]

=========================================== warnings summary ===========================================

============================= 25 passed, 2 skipped, 19 warnings in 12.07s ==============================
```

coverage.py provides also pretty HTML ouput page by module that indicate zone coverage. See in this directory

```console
quality/html_coverage
```
You can open it with a web navigator.


## Check type/annotation : mypy

TBD

# SonarQube in local

SonarQube allows you to visualize the results with a very nice layout. It's possible to install it in local.

## Installation with Docker

## Installation sonar-scanner

## Configuration SonarQube for grand

## Configuration to send code quality result to SonarQube

## Script update

```console
grand_quality_sonar
```

