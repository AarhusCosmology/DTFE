# Imports
import numpy as np
from nbodykit.lab import *
from nbodykit.io.binary import *
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import glob

chunk_size = 512

def read_chunk(filename, component=0):
    data = np.fromfile(filename, dtype=np.float32).reshape(-1, 3)
    return data[:, component].reshape((chunk_size, chunk_size, chunk_size))

def assemble_component(filepaths, full_shape, component=0):
    full_grid = np.zeros(full_shape, dtype=np.float32)

    def process_file(filepath):
        # Extract indices from filename
        # assuming filename like: p2048_b128_pm_a_1.00_v2_r_3_3_5.a_velVort
        base = os.path.basename(filepath)
        parts = base.split("_r_")[1].split(".")[0].split("_")
        x_idx, y_idx, z_idx = map(int, parts)
        print(f'Processing file, {filepath}', flush=True)
        return x_idx, y_idx, z_idx, read_chunk(filepath, component)

    with ThreadPoolExecutor() as executor:
        for x_idx, y_idx, z_idx, comp_data in executor.map(process_file, filepaths):
            xs = x_idx * chunk_size
            ys = y_idx * chunk_size
            zs = z_idx * chunk_size
            full_grid[xs:xs+chunk_size, ys:ys+chunk_size, zs:zs+chunk_size] = comp_data

    return full_grid






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
#filepath = '/home/camtgs/DTFE/output/'+filename

output = sys.argv[2]
outputpath = '/home/camtgs/DTFE/output/'+output

box_size = int(sys.argv[3])
grid_size = int(sys.argv[4])

N = int(grid_size/chunk_size)
full_shape = (N * chunk_size, N * chunk_size, N * chunk_size)

vort = []

#Data for powerspectra
#data = read_grad_field(filename, grid_size)
files = glob.glob(filename.replace('r_0_0_0', 'r_*_*_*'))

for component in range(3):
    w_grid = assemble_component(files, full_shape, component=component)

    vort_i = power_spectrum(w_grid, box_size)
#vorty = power_spectrum(data[1], box_size)
#vortz = power_spectrum(data[2], box_size)
    vort.append(vort_i['power'])

k = vort_i['k']
#    powerx = vortx['power']/(3e5**2)
#    powery = vorty['power']/(3e5**2)
#    powerz = vortz['power']/(3e5**2)
#power = vort/(3e5**2)
power = sum(vort)/(3e5**2)

#k, power = compute_vorticity(data, box_size)

np.savetxt(outputpath+'.txt', np.c_[k,power.real])
#np.savetxt(outputpath+'_x.txt', np.c_[k,powerx.real])
#np.savetxt(outputpath+'_y.txt', np.c_[k,powery.real])
#np.savetxt(outputpath+'_z.txt', np.c_[k,powerz.real])

