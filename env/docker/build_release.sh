#! /bin/bash

# parameter must be a git name tag/branch

tag=demo_master

docker rmi grandlib_release:$tag

#mkdir src4doc
#cd src4doc
git clone https://github.com/grand-mother/grand.git
cd grand
git checkout $1
cd ..
docker build -f ./release.dockerfile . --tag=grandlib_release:$tag

rm -rf grand
