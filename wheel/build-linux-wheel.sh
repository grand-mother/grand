#! /bin/bash

TAG="${1:-cp38-cp38}"
ARCH="${2:-x86_64}"

PYTHON="/opt/python/${TAG}/bin/python"
DOCKER_IMAGE="quay.io/pypa/manylinux1_${ARCH}"


PRE_CMD=
if [[ "${ARCH}" == "i686" ]]; then
    PRE_CMD=linux32
fi

SCRIPT=$(cat <<-END
${PRE_CMD}
set -ex

atexit() {
    chown -R ${uid}:${gid} /pwd/*
}
trap atexit EXIT

yum install -y zlib-devel

${PYTHON} -m pip install -U pip
${PYTHON} -m pip install cffi

cd /tmp
git clone -b v1.6.35 --depth 1 https://github.com/glennrp/libpng.git
cd libpng
./configure --prefix=/usr
make install

cd /pwd
make clean
make install PYTHON=${PYTHON}
GRAND_VERSION=${GRAND_VERSION} "${PYTHON}" wheel/setup.py bdist_wheel
export LD_LIBRARY_PATH="\$(pwd)/lib:\${LD_LIBRARY_PATH}"
auditwheel repair --plat=manylinux1_${ARCH} -w wheel dist/grand*.whl

echo "cleaning"
make clean
chmod a+rw wheel/grand*.whl
END
)

docker run --mount type=bind,source=$(pwd),target=/pwd \
           ${DOCKER_IMAGE} /bin/bash -c "${SCRIPT}"
