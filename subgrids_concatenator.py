# TO CONCATENATE ALL THE SUBGRIDS TO GET THE FULL GRID OF INTEGRALS

from scipy.optimize import curve_fit, minimize
from scipy import integrate
import emcee
import corner
import time
from multiprocessing import cpu_count, Pool
import os
import h5py 
import numpy as np
import astropy.table as aTable
from astropy.cosmology import Planck13
from astropy.cosmology import FlatLambdaCDM



def concatenate_subgrids(subgrids, n_division):
    n_subgrids = len(subgrids)
    strips_array = []
    bases_array = []
    bases2_array = []
    bases3_array = []
    bases4_array = []
    
    for i in range(0, n_subgrids, n_division):
        strips_array.append(np.concatenate((subgrids[i:i + n_division]), axis=0))
    
        
    for j in range(0, len(strips_array)-1, n_division):
        bases_array.append(np.concatenate((strips_array[j:j + n_division]), axis=1))
        
        
    for k in range(0, len(bases_array)-1, n_division):
        bases2_array.append(np.concatenate((bases_array[k:k + n_division]), axis=2))
        
        
    for a in range(0, len(bases2_array)-1, n_division):
        bases3_array.append(np.concatenate((bases2_array[a:a + n_division]), axis=3))
    
    
    for b in range(0, len(bases3_array)-1, n_division):
        bases4_array.append(np.concatenate((bases3_array[b:b + n_division]), axis=4))
        
        
    return np.concatenate(bases4_array, axis=5)


# MAIN

loaded_subgrids = np.load("subgrids_save_test.npy", allow_pickle=True)

n_division = 10
conc = concatenate_subgrids(loaded_subgrids, n_division)

np.save("conc_grid_6d_20sample.npy", conc)
print("The concatenated grid has been saved")

