#! /bin/bash

docker rmi grandlib_dev_eclipse

docker build -f dev_eclipse.dockerfile . --tag=grandlib_dev_eclipse
