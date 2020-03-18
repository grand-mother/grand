import os
import setuptools


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Programming Language :: C
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS
"""


with open("README.rst") as f:
    long_description = f.read() 


setuptools.setup(
    name="grand",
    version="0.0.1",
    maintainer="GRAND developers",
    maintainer_email="grand-dev@googlegroups.com",
    description="Core package for GRAND",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/grand-mother/grand",
    packages=setuptools.find_packages(exclude=("tests*",)),
    classifiers=[s for s in CLASSIFIERS.split(os.linesep) if s.strip()],
    license = "LGPLv3",
    platforms = ["Linux", "Mac OS-X", "Unix"],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"": ("_core.so", "libs/*.so", "libs/*.so.*",
        "libs/data/gull/*", "tools/data/egm96.png")},
    exclude_package_data = {"": ("tools/data/topography/*",)},
)
