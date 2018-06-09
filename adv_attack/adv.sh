#!/bin/sh
#$-q datasci
#$-q ddlab_test

cd research/experiment

python get_adv.py resnet resnet

python get_adv.py resnet densenet

python get_adv.py resnet vgg

python get_adv.py densenet resnet

python get_adv.py densenet densenet

python get_adv.py densenet vgg

python get_adv.py vgg resnet

python get_adv.py vgg densenet

python get_adv.py vgg vgg

