# A GRANDlib environment image, generated from this repository's own metadata.
#
# The point of difference from env/docker_amd64/ and env/docker_arm64/ is that
# nothing here is a second list to keep in step.  Those carry a
# requirements.txt of 53 unpinned package names, which was the only place
# declaring numba and lmfit and drifted from the conda environment for three
# years without anyone noticing.  This installs `.[dev]`, so the dependencies
# come from pyproject.toml -- the same declaration pip and the conda
# environment already use.  There is one source of truth and two ways to
# deliver it.
#
# ROOT is pinned to 6.36 to match env/conda/grand-dev.yml.  That matters: the
# published 2023 images carry 6.26, which takes a different branch in
# grand/dataio/descriptors.py, and while that branch does work (measured), a
# Docker user should not silently be running different code from everybody
# else.
#
# The image deliberately does NOT contain the ~1 GB data model.  env/setup.sh
# fetches it, compares versions and skips when current, so baking it in would
# quadruple the image to freeze something that changes on its own schedule.
# Fetch it at runtime, or mount a directory that already has it.
#
#     docker build -f env/docker/grandlib.dockerfile -t grandlib:dev .
#     docker run --rm -it -v "$PWD:/opt/grandlib" grandlib:dev
#     # then, inside:  source env/setup.sh && pytest tests/ -q

FROM rootproject/root:6.36.00-ubuntu25.04

# TURTLE and GULL are C libraries compiled by src/Makefile through
# env/setup.sh, so the toolchain has to be present even though nothing here is
# a C project.  git is needed because install_ext_lib.bash clones them.
# python3-pip because the rootproject/root image does not ship it: the build
# failed with "/usr/bin/python3: No module named pip" without it.  ROOT is
# built against the system interpreter in these images, so that is the one to
# install into rather than a separate virtualenv.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        git \
        ca-certificates \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/grandlib
COPY . /opt/grandlib

# `.[dev]` rather than a requirements file: the dependency list comes from
# pyproject.toml, which is what pip and the conda environment already use.
# Adding a second list here is precisely the mistake env/docker_amd64/ made.
#
# Ubuntu 25.04 marks the system Python externally managed (PEP 668).
# Overriding that is right in a container whose whole purpose *is* this Python
# environment, in a way it would not be on a workstation.
#
# This installs GRANDlib itself as well as its dependencies.  pyproject.toml
# does not build the C extension -- TURTLE and GULL are compiled by
# src/Makefile through env/setup.sh -- so the install is pure Python and the
# extension still has to be built at first run.
RUN python3 -m pip install --no-cache-dir --break-system-packages ".[dev]"

# env/setup.sh sets these itself, but having them present means `python3 -c
# "import grand"` works in a bare `docker run` with no sourcing.  A bind mount
# over /opt/grandlib shadows the installed copy, which is what you want when
# developing against the container.
ENV GRAND_ROOT=/opt/grandlib
ENV PYTHONPATH=/opt/grandlib

CMD ["/bin/bash"]
