import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

grid_size = int(sys.argv[3])
box_size = int(sys.argv[2])
folder = '/home/camtgs/DTFE/output/'
filename = sys.argv[1]
#type = sys.argv[4]

density_field = np.fromfile(folder + filename + '.a_den', dtype=np.float32).reshape(grid_size,grid_size,grid_size)
density_slice = density_field[:,:,:20]
density_data = density_slice.sum(2)/(density_slice.sum(2)).mean()

np.savetxt(folder + filename + '_den_data.txt', density_data)
