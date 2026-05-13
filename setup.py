from setuptools import setup
from setuptools.command.build_ext import build_ext
import subprocess
import shutil
import sys
from pathlib import Path


TURTLE_URL = "https://github.com/niess/turtle.git"
GULL_URL = "https://github.com/niess/gull.git"
TURTLE_TAG = "v0.8"
GULL_TAG = "286ace5"

ROOT_DIR = Path(__file__).parent
INSTALL_DIR = ROOT_DIR / "install"
BUILD_DIR = INSTALL_DIR / "zig-out"
SRC_DIR = ROOT_DIR / "src"

ZIG = "zig-0.16.0"


class CustomBuild(build_ext):
    def run(self):
        self.clone_repos()
        self.build_c_libs()
        self.build_python_ext()

    def clone_repos(self):
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        # turtle — clone specific tag directly
        turtle_dir = BUILD_DIR / "turtle"
        if not turtle_dir.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    TURTLE_TAG,
                    TURTLE_URL,
                    str(turtle_dir),
                ],
                check=True,
            )

        # gull — clone then fetch specific commit
        gull_dir = BUILD_DIR / "gull"
        if not gull_dir.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    GULL_URL,
                    str(gull_dir),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(gull_dir),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    GULL_TAG,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(gull_dir),
                    "checkout",
                    GULL_TAG,
                ],
                check=True,
            )

    def build_c_libs(self):
        if shutil.which("zig") is None:
            raise RuntimeError(
                "zig not found — please install Zig before running pip install. "
                "See https://ziglang.org/download/"
            )

        subprocess.run(
            [
                ZIG,
                "build",
                "-Doptimize=ReleaseFast",
            ],
            cwd=str(INSTALL_DIR),
            check=True,
        )

    def build_python_ext(self):
        subprocess.run(
            [
                sys.executable,
                str(SRC_DIR / "build_core.py"),
                str(BUILD_DIR),
            ],
            cwd=str(SRC_DIR),
            check=True,
        )


setup(
    cmdclass={"build_ext": CustomBuild},
)
