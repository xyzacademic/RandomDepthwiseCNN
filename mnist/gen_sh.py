import sys

data = []
data.append('#!/bin/sh\n')





gpu = 0

for foid in range(1, 11):
    i = 0

    k = 11
    for n in [1]:
        data.append('python stack_features.py %d %d %d %d %d\n\n'%(k, n, i, foid, gpu))
        i += 1




with open('generate_features.sh', 'w') as f:
    f.writelines(data)
