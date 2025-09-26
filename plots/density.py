import numpy as np
from nbodykit.lab import *
from nbodykit.io.binary import *
import sys
import os

filename = sys.argv[1]
filepath = '/home/camtgs/DTFE/output/'+filename

output = sys.argv[2]
outputpath = '/home/camtgs/DTFE/output/'+output

box_size = int(sys.argv[3])
grid_size = int(sys.argv[4])

data = np.fromfile(filepath, dtype=np.float32).reshape(grid_size,grid_size,grid_size)

mesh = ArrayMesh(data, BoxSize = box_size)
r = FFTPower(mesh, mode='1d')
den = r.power

k = den['k']
power = den['power']

np.savetxt(outputpath+'.txt', np.c_[k,power.real])
