#! /bin/bash

rm build/grand/_core.abi3.so
rm build/lib/*.so

make

cp build/grand/_core.abi3.so ../grand
cp build/lib/*.so ../lib
