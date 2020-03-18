#! /bin/bash

PYTHON="${1:-/opt/python/cp38-cp38/bin/python3.8}"
DOCKER_IMAGE="${2:-quay.io/pypa/manylinux1_x86_64}"


SCRIPT=$(cat <<-END
yum install -y zlib-devel

${PYTHON} -m pip install -U pip
${PYTHON} -m pip install cffi

cd /tmp
git clone -b v1.6.35 --depth 1 https://github.com/glennrp/libpng.git
cd libpng
./configure --prefix=/usr
make install

cd /work
make clean
make install PYTHON=${PYTHON}
"${PYTHON}" wheel/setup.py bdist_wheel
auditwheel repair dist/grand*.whl -L libs --plat=manylinux1_x86_64 -w wheel

echo "cleaning"
make clean
chmod a+rw wheel/grand*.whl
END
)

docker run --mount type=bind,source=$(pwd),target=/work \
           ${DOCKER_IMAGE} /bin/bash -c "${SCRIPT}"
