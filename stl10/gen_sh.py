import sys

data = []
data.append('#!/bin/sh\n')





gpu = 0

for foid in range(1, 11):
    i = 0

    k = 3
    for n in [7, 11, 15, 21]:
        data.append('python stack_features.py %d %d %d %d %d\n\n'%(k, n, i, foid, gpu))
        i += 1




with open('generate_features.sh', 'w') as f:
    f.writelines(data)
