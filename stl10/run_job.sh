#!/bin/sh

python stack_features.py --kernel-size 3 --layers 3 --num-features 100000 --batch-size 4 --seed 2018
python svm.py 3_3