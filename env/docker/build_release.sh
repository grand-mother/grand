#! /bin/bash

# parameter must be a git name tag/branch

tag=latest

docker rmi grandlib_release:$tag

git clone https://github.com/grand-mother/grand.git
cd grand
git checkout $1
cd ..

docker build -f ./release.dockerfile . --tag=grandlib_release:$tag

rm -rf grand
