#! /bin/bash

rm -f build/grand/_core.abi3.so
rm -f build/lib/*.so

make

cp build/grand/_core.abi3.so ../grand
cp build/lib/*.so ../lib
