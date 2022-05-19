#! /bin/bash

tag=0.1

cp ../../quality/requirements.txt requirements_qual.txt

docker build -f ci.dockerfile . --tag=grandlib_ci:$tag
