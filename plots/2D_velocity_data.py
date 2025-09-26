import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

def read_grad_field(filename, grid_size):
    """
    Read a vector field binary file with three components.
    """
    data = np.fromfile(filename, dtype=np.float32).reshape(grid_size**3,3)
    output = np.zeros((3, grid_size, grid_size, grid_size))
    for i in range(3):
        output[i] = data[:,i].reshape((grid_size, grid_size, grid_size))
    return output

grid_size = int(sys.argv[3])
box_size = int(sys.argv[2])
folder = '/home/camtgs/DTFE/output/'
filename = sys.argv[1]
a = float(sys.argv[4])
#type = sys.argv[4]

velocity_field = read_grad_field(folder + filename + '.a_vel', grid_size)
velocity_len = np.sqrt(velocity_field[0]**2 + velocity_field[1]**2 + velocity_field[2]**2)*np.sqrt(a)
velocity_slice = velocity_len[:,:,:20]
velocity_data = velocity_slice.mean(2)

np.savetxt(folder + filename + '_vel_data.txt', velocity_data)
