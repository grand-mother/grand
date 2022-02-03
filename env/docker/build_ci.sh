#! /bin/bash

tag=0.1

cp ../../quality/requirements.txt requirements_qual.txt

docker build -f Dockerfile_ci . --tag=grandlib_ci:$tag
