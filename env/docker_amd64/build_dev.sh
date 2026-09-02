#! /bin/bash

docker rmi grandlib_dev

cp ../../quality/requirements.txt requirements_qual.txt
# docs/apidoc-only/ was retired when the Sphinx tree was rebuilt; the
# documentation dependencies now live in docs/requirements.txt.
cp ../../docs/requirements.txt requirements_docs.txt

docker build -f dev.dockerfile . --tag=grandlib_dev
