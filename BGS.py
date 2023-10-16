from Data import Data
import numpy as np
from astropy.cosmology import Planck13


class BGS(Data):
    def __init__(self, file, is_bgs_bright=True):
        super().__init__(file)

        self.f_area = (173.641 / (4. * np.pi * (180 / np.pi) ** 2))
        self.is_bgs_bright = is_bgs_bright
        if is_bgs_bright:
            self.is_bgs = 'is_bgs_bright'
        else:
            self.is_bgs = 'is_bgs_faint'

        bgs_all = self.get_data()
        bgs = bgs_all[bgs_all[self.is_bgs]]
        self.z = bgs['Z_HP'].data
        self.x = bgs['provabgs_logMstar'].data
        self.x_median = np.median(self.x, axis=1)
        self.w_zfail = bgs['provabgs_w_zfail'].data
        self.w_fib = bgs['provabgs_w_fibassign'].data
        self.w_spec = self.w_zfail * self.w_fib
        self.vmax = bgs['Vmax'].data

        # Setting attributes for the mass completeness limit parameters
        self.f = -1.34199453
        self.b = 13.90578909
        self.c = 8.53522654

    def select_galaxies(self, zmin, zmax):
        """Return z, x, x_median, w_spec, vmax for galaxies between (zmin,zmax)"""
        mask_zlim = (self.z > zmin) & (self.z < zmax) & (self.x_median > self.mass_completeness_limit(self.z))

        v_zmin = Planck13.comoving_volume(zmin).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3
        v_zmax = Planck13.comoving_volume(zmax).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3

        z = self.z[mask_zlim]
        x = self.x[mask_zlim]
        x_median = self.x_median[mask_zlim]
        w_zfail = self.w_zfail[mask_zlim]
        w_fib = self.w_fib[mask_zlim]
        w_spec = self.w_spec[mask_zlim]
        vmax = (self.vmax.clip(v_zmin, v_zmax) - v_zmin)[mask_zlim]

        return z, x, x_median, w_spec, vmax

    def mass_completeness_limit(self, z):
        return 4 * np.pi * self.f * z ** 2 + self.b * z + self.c

    def get_number_galaxies(self):
        if self.z.shape[0] == self.x.shape[0]:
            return self.z.shape[0]
