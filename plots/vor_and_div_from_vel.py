import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
from nbodykit.lab import *
from nbodykit.io.binary import *
from nbodykit.io.gadget import *
import sys
import os

def read_grad_field(filename, grid_size):
    """
    Read a vector field binary file with three components.
    """
    data = np.fromfile(filename, dtype=np.float32).reshape(grid_size**3,3)
    output = np.zeros((3, grid_size, grid_size, grid_size))
    for i in range(3):
        output[i] = data[:,i].reshape((grid_size, grid_size, grid_size))
    return output

def vorticity_from_velocity(vx, vy, vz, boxsize):
    # compute Fourier transform of velocity field vx, vy, vz
    velocity_x = fftn(vx)
    velocity_y = fftn(vy)
    velocity_z = fftn(vz)

    # compute k-space grid with correct scaling
    grid_size_k = velocity_x.shape[0]

    k_vals = fftfreq(grid_size_k, d=boxsize/grid_size_k)*2*np.pi
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[0,0,0] = 1
    
    velocity_x[0,0,0] = 0
    velocity_y[0,0,0] = 0
    velocity_z[0,0,0] = 0

    # compute vorticities in k-space
    omega_23 = np.real(ifftn(0.5 * 1j * (ky * velocity_z - kz * velocity_y)))
    omega_13 = np.real(ifftn(-0.5 * 1j * (kz * velocity_x - kx * velocity_z)))
    omega_12 = np.real(ifftn(0.5 * 1j * (kx * velocity_y - ky * velocity_x)))

    return omega_12, omega_13, omega_23


def divergence_from_velocity(vx, vy, vz, boxsize):
    # check fftn from nbodykit
    # compute fourier transform of velocity field vx vy vz
    velocity_x = fftn(vx)
    velocity_y = fftn(vy)
    velocity_z = fftn(vz)

    # compute divergence in k-space
    grid_size_k = velocity_x.shape[0]  
    
    k_vals = fftfreq(grid_size_k, d=boxsize/grid_size_k)*2*np.pi
    kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
#    k_squared = kx**2 + ky**2 + kz**2
#    k_squared[0,0,0] = 1
#    k_mag = np.sqrt(k_squared)
    
    velocity_x[0,0,0] = 0
    velocity_y[0,0,0] = 0
    velocity_z[0,0,0] = 0

    div_k = 1j*(kx*velocity_x + ky*velocity_y + kz*velocity_z)
    div = np.real(ifftn(div_k))

#    power = np.real(np.abs(div_k)**2)

    return div

def power_spectrum(data, box_size):
    """
    Compute the power spectrum of a scalar field.
    """
    mesh = ArrayMesh(data, BoxSize = box_size)
    r = FFTPower(mesh, mode='1d')
    power = r.power
    return power

def compute_vorticity(omega_12, omega_13, omega_23, box_size):
    vort_12 = power_spectrum(omega_12, box_size)
    vort_13 = power_spectrum(omega_13, box_size)
    vort_23 = power_spectrum(omega_23, box_size)
    vort = vort_12['power'] + vort_13['power'] + vort_23['power']
    
    k = vort_12['k']
#    power_12 = vort_12['power']/(3e5**2)
#    power_13 = vort_13['power']/(3e5**2)
#    power_23 = vort_23['power']/(3e5**2)
    power = vort/(9e4)

    return k, power

def compute_divergence(div, box_size):
    div_power = power_spectrum(div, box_size)
    
    k = div_power['k']
    power = div_power['power']/(9e4)

    return k, power

def divergence_power_spectrum(vx, vy, vz, box_size):
    div = divergence_from_velocity(vx, vy, vz, box_size)
    k, power = compute_divergence(div, box_size)
    return k, power

def vorticity_power_spectrum(vx, vy, vz, box_size):
    omega_12, omega_13, omega_23 = vorticity_from_velocity(vx, vy, vz, box_size)
    k, power = compute_vorticity(omega_12, omega_13, omega_23, box_size)
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

kdiv, powerdiv = divergence_power_spectrum(data[0], data[1], data[2], box_size)

kvor, powervor = vorticity_power_spectrum(data[0], data[1], data[2], box_size)

np.savetxt(outputpath+'_vel_div.txt', np.c_[kdiv,powerdiv.real])
np.savetxt(outputpath+'_vel_vor.txt', np.c_[kvor,powervor.real])

