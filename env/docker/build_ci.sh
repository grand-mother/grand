#! /bin/bash

tag=0.1


cp ../../quality/requirements.txt requirements_qual.txt


docker build -f Dockerfile_ci . --tag=grandlib_ci:$tag


#docker tag grand_env:$tag jcolley/grandlib_ci:$tag
#docker push jcolley/grandlib_ci:$tag