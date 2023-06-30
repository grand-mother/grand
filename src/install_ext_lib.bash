#! /bin/bash

#make clean
make

cp build/grand/_core.abi3.so ../grand
cp build/lib/*.so ../lib
