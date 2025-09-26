# Imports
import numpy as np
from nbodykit.lab import *
from nbodykit.io.binary import *
import sys
import os

#Functions
# Vigtigt!!! Hastigheden er med stor sandsynlighed i km/s!!!
#Dvs. alle powerspektre skal ganges med en faktor 1/c^2, hvor c = 3e5 km/s

def read_grad_field(filename, grid_size):
    """
    Read a vector field binary file with three components.
    """
    data = np.fromfile(filename, dtype=np.float32).reshape(grid_size**3,3)
    output = np.zeros((3, grid_size, grid_size, grid_size))
    for i in range(3):
        output[i] = data[:,i].reshape((grid_size, grid_size, grid_size))
    return output

def power_spectrum(data, box_size):
    """
    Compute the power spectrum of a scalar field.
    """
    mesh = ArrayMesh(data, BoxSize = box_size)
    r = FFTPower(mesh, mode='1d')
    power = r.power
    return power

def compute_vorticity(data, box_size):
    vortx = power_spectrum(data[0], box_size)
    vorty = power_spectrum(data[1], box_size)
    vortz = power_spectrum(data[2], box_size)
    vort = vortx['power'] + vorty['power'] + vortz['power']
    
    k = vortx['k']
#    powerx = vortx['power']/(3e5**2)
#    powery = vorty['power']/(3e5**2)
#    powerz = vortz['power']/(3e5**2)
    power = vort/(3e5**2)

    return k, power

#Files
filename = sys.argv[1]
filepath = '/home/camtgs/DTFE/output/'+filename

output = sys.argv[2]
outputpath = '/home/camtgs/DTFE/output/'+output

box_size = int(sys.argv[3])
grid_size = int(sys.argv[4])

#Data for powerspectra
data = read_grad_field(filepath, grid_size)

k, power = compute_vorticity(data, box_size)

np.savetxt(outputpath+'.txt', np.c_[k,power.real])
#np.savetxt(outputpath+'_x.txt', np.c_[k,powerx.real])
#np.savetxt(outputpath+'_y.txt', np.c_[k,powery.real])
#np.savetxt(outputpath+'_z.txt', np.c_[k,powerz.real])

