# Building and installing GRANDlib

To intall GRANDlib, we recommend using virtual environments. The following example showcase
`conda` virtual environment but similar command exist for `venv`.
To do so, execute the following commands in the terminal:

```bash
conda env create -f grandlib_environment.yaml
conda activate grand
```

These two commands install all the packages and library needed by GRANDlib,
however, GRANDlib itself is not installed as is depends on C libraries.

The compilation of these C library, their integration into the GRANDlib library and its
installation are done by a python script which can be run using the following command:


```bash
python install.py
```

---
**Note**

Make sure that you are within the conda grand environment before you run the `install.py` script.

---

This script calls the [Zig](https://ziglang.org/) compiler to compile the two C libraries,
[TURTLE](https://github.com/niess/turtle) and [GULL](https://github.com/niess/gull). If a Zig
compiler of the required version (0.16.0+) is not found, the script automatically download and
install the relevant Zig compiler in the `.zig` folder of this directory -- after having check
the signature of the downloaded file.

The downloading of the Zig compiler can be skipped with the following command:

```bash
python install.py --no-download
```

Moreover, a path to a zig compiler binary can be specify with the following command:

```bash
python install.py --zig <PATH_TO_ZIG_BINARY>
```
