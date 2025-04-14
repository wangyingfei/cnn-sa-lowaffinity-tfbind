import h5py 
import numpy as np 
from collections import OrderedDict 
import sys 
import os
    
def YRcoding(s):
    encoded = ''.join([{'A':'R', 'T':'Y', 'C':'Y','G':'R', 'n':'n'}[B] for B in s])
    return encoded
    
input_file = sys.argv[1]

out_file = sys.argv[2]
out = open(out_file, "w")
for line in open(input_file, "r").readlines():
    out.write(YRcoding(line.rstrip().split("\t")[0]) + "\n")

out.close()


