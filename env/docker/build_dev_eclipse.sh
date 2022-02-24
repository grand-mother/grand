#! /bin/bash

tag=latest

#docker rmi grandlib_dev_eclipse:$tag

docker build -f dev_eclipse.dockerfile . --tag=grandlib_dev_eclipse:$tag
