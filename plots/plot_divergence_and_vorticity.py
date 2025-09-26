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
    data = np.fromfile(filename, dtype=np.float32).reshape(grid_size**3,3,3)
    output = np.zeros((3, 3, grid_size, grid_size, grid_size))
    for i in range(3):
        for j in range(3):
            output[i, j] = data[:,i,j].reshape((grid_size, grid_size, grid_size))
    return output

def power_spectrum(data, box_size):
    """
    Compute the power spectrum of a scalar field.
    """
    mesh = ArrayMesh(data, BoxSize = box_size)
    r = FFTPower(mesh, mode='1d')
    power = r.power
    return power

def compute_divergence(data, box_size):
    grad = power_spectrum(data[0,0]+data[1,1]+data[2,2], box_size)
    
    k = grad['k']
    power = grad['power']/(3e5**2)

    return k, power

def compute_vorticity(data, box_size):
    vortx = power_spectrum(0.5*(data[2,1]-data[1,2]), box_size)
    vorty = power_spectrum(0.5*(data[0,2]-data[2,0]), box_size)
    vortz = power_spectrum(0.5*(data[1,0]-data[0,1]), box_size)
    vort = vortx['power'] + vorty['power'] + vortz['power']
    
    k = vortx['k']
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

kdiv, powerdiv = compute_divergence(data, box_size)

kvor, powervor = compute_vorticity(data, box_size)

np.savetxt(outputpath+'_div.txt', np.c_[kdiv,powerdiv.real])
np.savetxt(outputpath+'_vor.txt', np.c_[kvor,powervor.real])
