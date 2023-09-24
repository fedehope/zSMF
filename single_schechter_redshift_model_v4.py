# SINGLE SCHECHTER REDSHIFT MODEL V4

from scipy.optimize import curve_fit, minimize
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
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





# def region_limits(z_i, m_i):
#     if m_i > 11.401850153336317:
#         threshold = 0.4
#         return np.array([
#         [0.01, mass_completeness_limit(z_i)],
#         [z_i, mass_completeness_limit(z_i)],
#         [threshold, mass_completeness_limit(threshold)],
#         [threshold, 13.],
#         [0.01, 13.]])
#     return np.array([
#         [0.01, mass_completeness_limit(z_i)],
#         [z_i, mass_completeness_limit(z_i)],
#         [zmax_mass_completeness_limit(m_i), m_i],
#         [zmax_mass_completeness_limit(m_i), 13.],
#         [0.01, 13]])





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
            [mass_completeness_limit, mmax_thr]])
    
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




def integral_calculation(limits, a0, a1, a2, a3):
    result = 0
    result += integrate.dblquad(smf_single_schechter_integral, limits[0][0],
                                limits[0][1], limits[1][0], limits[1][1], args=(a0, a1, a2, a3))[0]
    result += integrate.dblquad(smf_single_schechter_integral, limits[2][0],
                            limits[2][1], limits[3][0] , limits[3][1], args=(a0, a1, a2, a3))[0]
    
    return result





def log_likelihood(a0, a1, a2, a3, interp, x_median, w, z, x):
    q = smf_single_schechter_sty(x, z, a0, a1, a2, a3)
    
    if 9.5 < a0 < 13.5 and \
        2.5 < a1 < 6. and \
       -2.5 < a2 < -0.5  and \
       0. < a3 < 3.:
        a0_v = np.repeat(a0, x_median.shape[0])
        a1_v = np.repeat(a1, x_median.shape[0])
        a2_v = np.repeat(a2, x_median.shape[0])
        a3_v = np.repeat(a3, x_median.shape[0])

        pt = np.array([a0_v, a1_v, a2_v, a3_v, x_median, z]).T
        I = interp(pt)

        a = np.log10(np.sum(q, axis=1)) - np.log10(I)
        return a * w
    
    return -np.inf




def log_prior(theta):
    a0, a1, a2, a3 = theta
    if 9.5 < a0 < 13.5 and \
        2.5 < a1 < 6. and \
       -2.5 < a2 < -0.5  and \
       0. < a3 < 3.:
        return 0
    return -np.inf



# Prior based on the grid bounderies (6d - 10 point/dim)
# def log_prior(theta):
#     a0, a1, a2, a3 = theta
#     if 9. < a0 < 13. and \
#         0. < a1 < 0.5 and \
#        -3 < a2 < 0.  and \
#        2. < a3 < 4.:
#         return 0
#     return -np.inf





def posterior(theta, interp, x_median, w, z, x):
    a0, a1, a2, a3 = theta
   
    l = log_likelihood(a0, a1, a2, a3, interp, x_median, w, z, x)
    return log_prior(theta) + np.sum(l)


# def posterior(theta, integral_limits_list, w, z, x):
#     a0, a1, a2, a3 = theta
#     I = []
#     for limit in integral_limits_list:
#         I.append(integral_calculation(limit, a0, a1, a2, a3))
#     l = log_likelihood(a0, a1, a2, a3, I, w, z, x)
#     return log_prior(theta) + np.sum(l)



# -------------- MAIN -------------------------#

# Define cosmology
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307)

# Read hdf5 for BGS data
bgs = aTable.Table.read('BGS_ANY_full.provabgs.lite.hdf5')
is_bgs_bright = bgs['is_bgs_bright']
is_bgs_faint = bgs['is_bgs_faint']

bgs = bgs[bgs['is_bgs_bright']]

# Gathering data
# Exluding galaxies for z < 0.01 and z > 0.21 because I don't have the mass completeness limit for those
mask_zlim = (bgs['Z_HP'].data > 0.01) & (bgs['Z_HP'].data < 0.4)

z_tot = bgs['Z_HP'].data[mask_zlim]
x_tot = bgs['provabgs_logMstar'].data[mask_zlim]
x_median_tot = np.median(x_tot, axis=1)
w_zfail_tot = bgs['provabgs_w_zfail'].data[mask_zlim]
w_fib_tot = bgs['provabgs_w_fibassign'].data[mask_zlim]
vmax_tot = bgs['Vmax'].data[mask_zlim]


mass_comp_lim = mass_completeness_limit(z_tot)
mask_mlim = []
for i in range(len(x_median_tot)):
    mask_mlim.append(x_median_tot[i] > mass_comp_lim[i])
    

mask = (w_zfail_tot > 0) & (mask_mlim)

z = z_tot[mask].astype(np.float32)
x = x_tot[mask].astype(np.float32)
x_median = x_median_tot[mask].astype(np.float32)
w_zfail = w_zfail_tot[mask].astype(np.float32)
w_fib = w_fib_tot[mask].astype(np.float32)
vmax = vmax_tot[mask].astype(np.float32)


f_area = (173.641/(4.*np.pi*(180/np.pi)**2))
v_zmin = Planck13.comoving_volume(0.01).value * Planck13.h**3 * f_area # (Mpc/h)^3
v_zmax = Planck13.comoving_volume(0.09).value * Planck13.h**3 * f_area # (Mpc/h)^3
v_sub = v_zmax - v_zmin


# w_spec * 1/Vmax
w = (w_zfail*w_fib) * v_sub / (vmax.clip(v_zmin, v_zmax) - v_zmin)
n = np.sum(w)/v_sub

# Spectroscopic weights
w_spec = (w_zfail*w_fib)


# Load the 6D-grid of integrals and interpolate it
nn = 20
a0_lin = np.linspace(9.5, 13.5, nn)
a1_lin = np.linspace(2.5, 6., nn)
a2_lin = np.linspace(-2.5, -0.5, nn)
a3_lin = np.linspace(0, 3., nn)
m_lin = np.linspace(x_median.min(), x_median.max(), nn)
z_lin = np.linspace(z.min(), z.max(), nn)

grid_data = np.load('conc_grid_6d_20sample.npy')
interp = RegularGridInterpolator((a0_lin, a1_lin, a2_lin, a3_lin, m_lin, z_lin), grid_data)


# Integral limits for likelihood
# integral_limits_list = []
# for z_i, m_i in zip(z, x_median):
#     integral_limits_list.append(integral_limits(z_i, m_i))

#  Emcee
a0, a1, a2, a3 = 10., 3.2, -1.5, 1.5

# start_time = time.time()
# # I = []
# # for limit in integral_limits_list:
# #     I.append(integral_calculation(limit, a0, a1, a2, a3))
    
# integ = np.array(I)

# end_time = time.time()
# print("Integral array calc took {0:.1f} minutes".format((end_time-start_time)/60.))
# print(f" for {x.shape[0]} galaxies")

theta = [a0, a1, a2, a3]
args = (interp, x_median, w_spec, z, x)

start = np.array(theta)
nwalkers = 50
nstep = 600
pos = start + 1e-4 * np.random.randn(nwalkers, 4)
nwalkers, ndim = pos.shape

# Initialize the backend to save the chain
filename = "redshift_fit_1Schechter_2_50w_600s_v4_newgrid.h5"
backend = emcee.backends.HDFBackend(filename)



run_emcee = True



if run_emcee:
    with Pool() as pool:
        backend.reset(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=args, backend=backend, pool=pool)
        start_time = time.time()
        sampler.run_mcmc(pos, nstep, progress=True);
        end_time = time.time()
        print("Multiprocessing took {0:.1f} minutes".format((end_time-start_time)/60.))