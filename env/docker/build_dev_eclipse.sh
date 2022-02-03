#! /bin/bash

tag=latest

docker rmi grandlib_dev_eclipse:$tag

docker build -f Dockerfile_dev_eclispe . --tag=grandlib_dev_eclipse:$tag
