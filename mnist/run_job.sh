#!/bin/sh

python stack_features.py --kernel-size 5 --layers 1 --num-features 100000 --batch-size 8 --seed 2018
python svm.py 5_1