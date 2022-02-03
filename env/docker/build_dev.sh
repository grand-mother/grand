#! /bin/bash


tag=latest

docker rmi grandlib_dev:$tag

cp ../../quality/requirements.txt requirements_qual.txt
cp ../../docs/apidoc-only/doxygen-rtd/requirements.txt requirements_docs.txt


docker build -f Dockerfile_dev2 . --tag=grandlib_dev:$tag


#docker tag grand_env:$tag jcolley/grandlib_ci:$tag
#docker push jcolley/grandlib_ci:$tag