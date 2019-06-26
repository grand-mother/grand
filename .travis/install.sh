#! /bin/bash

set -ex


install_packages() {
    pip install codecov mypy
}


install_stubs() {
    git clone https://github.com/numpy/numpy-stubs.git /tmp/numpy-stubs
    mkdir -p user/grand/stubs
    mv /tmp/numpy-stubs/numpy-stubs user/grand/stubs/numpy
}


install_packages
install_stubs
