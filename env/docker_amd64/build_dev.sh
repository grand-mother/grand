#! /bin/bash

docker rmi grandlib_dev

cp ../../quality/requirements.txt requirements_qual.txt
cp ../../docs/apidoc-only/doxygen-rtd/requirements.txt requirements_docs.txt

docker build -f dev.dockerfile . --tag=grandlib_dev
