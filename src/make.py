from cffi import FFI
import os
from pathlib import Path
import platform
from pycparser import parse_file, c_generator
import sys


SRC_DIR = Path(__file__).parent.resolve()
try:
    BUILD_DIR = Path(sys.argv[1])
except IndexError:
    BUILD_DIR = Path("build")
BUILD_DIR = BUILD_DIR.resolve()

LIB_DIR = BUILD_DIR / "grand/libs"
INC_DIR = BUILD_DIR / "include"
TMP_DIR = BUILD_DIR / "tmp"
PACKAGE_PATH = BUILD_DIR / "grand"


ffibuilder = FFI()


def include(path, **opts):
    args = []
    for k, v in opts.items():
        args.append(f"-D{k}={v}")
    args.append("-I" + str(INC_DIR))
    args.append("-DFILE=struct FILE")

    ast = parse_file(str(path), use_cpp = True, cpp_args = args)
    generator = c_generator.CGenerator()
    header = generator.visit(ast)
    ffibuilder.cdef(header)

include(SRC_DIR / "grand.h")


def build():
    if platform.system() == "Darwin":
        rpath = rpath = ["-Wl,-rpath,@loader_path/libs"]
    else:
        rpath = ["-Wl,-rpath,$ORIGIN/libs"]

    with open(SRC_DIR / "grand.c") as f:
        ffibuilder.set_source("_core",
            f.read(),
            libraries = ["turtle", "gull"],
            include_dirs = [str(INC_DIR), str(SRC_DIR)],
            library_dirs = [str(LIB_DIR)],
            extra_link_args = rpath
        )

    TMP_DIR.mkdir(parents = True, exist_ok = True)
    PACKAGE_PATH.mkdir(parents = True, exist_ok = True)

    os.chdir(TMP_DIR)
    module = Path(ffibuilder.compile(verbose=False))
    module = module.rename(PACKAGE_PATH / "_core.so")


if __name__ == "__main__":
    build()

