#! /bin/bash

# parameter must be a git name tag/branch


docker rmi grandlib_release

git clone https://github.com/grand-mother/grand.git
cd grand
git checkout $1
cd ..

docker build -f ./release.dockerfile . --tag=grandlib_release

rm -rf grand
