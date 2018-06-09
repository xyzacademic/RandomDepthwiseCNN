#!/bin/sh

#$-q datasci
#$-q ddlab_test

cd research/experiment

python stl10.py densenet 0 0
python stl10.py densenet 1 1
