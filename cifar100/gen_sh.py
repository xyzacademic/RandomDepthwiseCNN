import sys

data = []
data.append('#!/bin/sh\n')





gpu = 0

for foid in range(1, 11):
    i = 0

    k = 3
    for n in range(3,10):
        data.append('python stack_features.py %d %d %d %d %d\n\n'%(k, n, i, foid, gpu))
        i += 1

    k = 5

    for n in range(2,5):
        data.append('python stack_features.py %d %d %d %d %d\n\n'%(k, n, i, foid, gpu))
        i += 1


with open('generate_features.sh', 'w') as f:
    f.writelines(data)
