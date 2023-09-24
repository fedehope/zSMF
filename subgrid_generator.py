# SINGLE SCHECHTER REDSHIFT MODEL V2

from scipy.optimize import curve_fit, minimize
from scipy.interpolate import RegularGridInterpolator
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



def mass_completeness_limit(z):
    f, b, c = [-1.34199453, 13.90578909,  8.53522654]
    return 4*np.pi*f*z**2 + b*z + c


def zmax_mass_completeness_limit(m):
    f, b, c = [-1.34199453, 13.90578909, 8.53522654]

    discriminant = -16*np.pi*f*c + 16*np.pi*f*m + b**2
    sqrt_discriminant = np.sqrt(discriminant)

    z1 = (-b + sqrt_discriminant) / (8 * np.pi * f)
    
    return z1


def region_limits(z_i, m_i):
    if m_i > 11.401850153336317:
        threshold = 0.4
        return np.array([
        [0.01, mass_completeness_limit(z_i)],
        [z_i, mass_completeness_limit(z_i)],
        [threshold, mass_completeness_limit(threshold)],
        [threshold, 13.],
        [0.01, 13.]])
    return np.array([
        [0.01, mass_completeness_limit(z_i)],
        [z_i, mass_completeness_limit(z_i)],
        [zmax_mass_completeness_limit(m_i), m_i],
        [zmax_mass_completeness_limit(m_i), 13.],
        [0.01, 13]])



def integral_limits(z_i, m_i):
    
    zmax_thr = 0.4
    mmax_thr = 13.
    m_lim_i = mass_completeness_limit(z_i)
    zmax_thr_i = zmax_mass_completeness_limit(mass_completeness_limit(zmax_thr))
    
    if m_i > 11.401850153336317:
        return np.array([
            [0.01, z_i],
            [m_lim_i, mmax_thr],
            [z_i, zmax_thr_i],
            [mass_completeness_limit ,mmax_thr]])
    
    z_max_i = zmax_mass_completeness_limit(m_i)
    
    return np.array([
        [0.01, z_i],
        [m_lim_i, mmax_thr],
        [z_i, z_max_i],
        [mass_completeness_limit, mmax_thr]])

    

    
def smf_single_schechter_sty(x, z, a0, a1, a2, a3):
    logM = a0 + a1*z
    alpha1 = a2 + a3*z
    
    term0 = np.exp(-10 ** (x-logM[:,None]))
    term1 = 10 ** ((alpha1+1)[:,None]*(x - logM[:,None]))
    return term0 * term1



def smf_single_schechter_integral(x, z, a0, a1, a2, a3):
    logM = a0 + a1*z
    alpha1 = a2 + a3*z
    term0 = np.exp(-10 ** (x-logM))
    term1 = 10 ** ((alpha1+1)*(x - logM))
    return term0 * term1



def log_likelihood(a0, a1, a2, a3, I, w, z, x):
    q = smf_single_schechter_sty(x, z, a0, a1, a2, a3)
    a = np.log10(np.sum(q, axis=1)) - np.log10(np.array(I))
    return a * w



def integral_calculation(limits, a0, a1, a2, a3):
    result = 0
    result += integrate.dblquad(smf_single_schechter_integral, limits[0][0],
                                limits[0][1], limits[1][0], limits[1][1], args=(a0, a1, a2, a3), 
                                epsabs=1e-3, epsrel=1e-3)[0]
    result += integrate.dblquad(smf_single_schechter_integral, limits[2][0],
                            limits[2][1], limits[3][0] , limits[3][1], args=(a0, a1, a2, a3), 
                                epsabs=1e-3, epsrel=1e-3)[0]
    
    return result




def integral_calculation2(a0, a1, a2, a3, m, z):
    limits = integral_limits(z, m)
    result = 0
    result += integrate.dblquad(smf_single_schechter_integral, limits[0][0],
                                limits[0][1], limits[1][0], limits[1][1], args=(a0, a1, a2, a3), 
                                epsabs=1e-3, epsrel=1e-3)[0]
    result += integrate.dblquad(smf_single_schechter_integral, limits[2][0],
                            limits[2][1], limits[3][0] , limits[3][1], args=(a0, a1, a2, a3), 
                                epsabs=1e-3, epsrel=1e-3)[0]
    
    return result




def log_prior(theta):
    a0, a1, a2, a3 = theta
    if 9.5 < a0 < 13.5 and \
        -4. < a1 < 4. and \
       -2.5 < a2 < -0.5  and \
       1.5 < a3 < 6.:
        return 0
    return -np.inf



def posterior(theta, integral_limits_list, w, z, x):
    a0, a1, a2, a3 = theta
    I = []
    for limit in integral_limits_list:
        I.append(integral_calculation(limit, a0, a1, a2, a3))
        
    l = log_likelihood(a0, a1, a2, a3, I, w, z, x)
    return log_prior(theta) + np.sum(l)


# -------------- MAIN -------------------------#

# Define cosmology
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307)

# Read hdf5 for BGS data
bgs = aTable.Table.read('BGS_ANY_full.provabgs.lite.hdf5')
is_bgs_bright = bgs['is_bgs_bright']
is_bgs_faint = bgs['is_bgs_faint']

bgs = bgs[bgs['is_bgs_bright']]

mask_zlim = (bgs['Z_HP'].data > 0.01)

z_tot = bgs['Z_HP'].data[mask_zlim]
x_tot = bgs['provabgs_logMstar'].data[mask_zlim]
x_median_tot = np.median(x_tot, axis=1)
w_zfail_tot = bgs['provabgs_w_zfail'].data[mask_zlim]
w_fib_tot = bgs['provabgs_w_fibassign'].data[mask_zlim]
vmax_tot = bgs['Vmax'].data[mask_zlim]

mask_mlim = []
for i in range(len(x_median_tot)):
    mask_mlim.append(x_median_tot[i] > mass_completeness_limit(z_tot[i]))
    

mask = (w_zfail_tot > 0) & (mask_mlim) & (z_tot<0.4)

z = z_tot[mask].astype(np.float32)
x = x_tot[mask].astype(np.float32)
x_median = x_median_tot[mask].astype(np.float32)
w_zfail = w_zfail_tot[mask].astype(np.float32)
w_fib = w_fib_tot[mask].astype(np.float32)
vmax = vmax_tot[mask].astype(np.float32)

# Spectroscopic weights
w_spec = (w_zfail*w_fib)

# 6D GRID  (a0, a1, a2, a3, m, z)


# point per dimension
nn = 30

a0_lin = np.linspace(9.5, 13.5, nn)
a1_lin = np.linspace(2.5, 6., nn)
a2_lin = np.linspace(-2.5, -0.5, nn)
a3_lin = np.linspace(0, 3., nn)
m_lin = np.linspace(x_median.min(), x_median.max(), nn)
z_lin = np.linspace(z.min(), z.max(), nn)

grid_a0, grid_a1, grid_a2, grid_a3, grid_m, grid_z = np.meshgrid(a0_lin, a1_lin, a2_lin, a3_lin, m_lin, z_lin, indexing='ij')


start_time = time.time()

# Define the subgrid size
n_division = 10
subgrid_size = 3

# Divide the cube meshgrid into eight adjacent subgrids
subgrids = []

for o in range(n_division):
    for s in range(n_division):
        for q in range(n_division):
            for k in range(n_division):
                for j in range(n_division):
                    for i in range(n_division):
                        subgrid_a0 = grid_a0[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrid_a1 = grid_a1[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrid_a2 = grid_a2[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrid_a3 = grid_a3[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrid_m = grid_m[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrid_z = grid_z[i*subgrid_size:(i+1)*subgrid_size, j*subgrid_size:(j+1)*subgrid_size,
                                             k*subgrid_size:(k+1)*subgrid_size, q*subgrid_size:(q+1)*subgrid_size,
                                             s*subgrid_size:(s+1)*subgrid_size, o*subgrid_size:(o+1)*subgrid_size]
                        subgrids.append((subgrid_a0, subgrid_a1, subgrid_a2, subgrid_a3, subgrid_m, subgrid_z))
                        
                        
                        
results_subgrids = []

for subgrid in subgrids:
    results_subgrid = np.zeros((subgrid[0][0].shape[0], subgrid[0][0].shape[0],
                                subgrid[0][0].shape[0], subgrid[0][0].shape[0],
                                subgrid[0][0].shape[0], subgrid[0][0].shape[0]))
    
    for o in range(subgrid_size):
        for s in range(subgrid_size):
            for q in range(subgrid_size):
                for k in range(subgrid_size):
                    for j in range(subgrid_size):
                        for i in range(subgrid_size):
                            results_subgrid[i,j,k,q,s,o] = integral_calculation2(subgrid[0][i,j,k,q,s,o], 
                                                                                 subgrid[1][i,j,k,q,s,o],
                                                                                 subgrid[2][i,j,k,q,s,o],
                                                                                 subgrid[3][i,j,k,q,s,o],
                                                                                 subgrid[4][i,j,k,q,s,o],
                                                                                 subgrid[5][i,j,k,q,s,o])
                            

    results_subgrids.append(results_subgrid)
    


end_time = time.time()

print(f"{nn} sample " + " took {0:.1f} min ".format((end_time-start_time)/60.))
print('n_division = ', n_division, 'subgrid_size = ', subgrid_size)
np.save(f"subgrids_{nn}sample.npy", results_subgrids)