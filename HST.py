import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

from Data import Data


class HST(Data):
    def __init__(self, file):
        super().__init__(file)
        self.footprint = 0.25  # deg^2
        self.sky = 41253  # deg^2
        self.cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307)

        hst_all = self.get_data()
        self.z = hst_all['z_peak'].data
        self.x = hst_all['lmass'].data
        self.vmax = None

    def select_galaxies(self, zmin, zmax):
        """Return z, x, x_median, w_spec, vmax for galaxies between (zmin,zmax)"""
        mask_zlim = (self.z > zmin) & (self.z < zmax)

        z = self.z[mask_zlim]
        x = self.x[mask_zlim]
        mask_mlim = (x > self.mass_completeness_limit(z))

        return z[mask_mlim], x[mask_mlim]

    def mass_completeness_limit(self, z):

        x_values = [0.65, 1.0, 1.5, 2.1, 3.0]
        y_values = [8.72, 9.07, 9.63, 9.79, 10.15]

        f = interp1d(x_values, y_values)
        return f(z)

    def zmax_lim(self, mstar):
        y_values = [8.72, 9.07, 9.63, 9.79, 10.15]
        x_values = [0.65, 1.0, 1.5, 2.1, 3.0]

        f = interp1d(y_values, x_values)
        if mstar > y_values[-1]:
            return x_values[-1]
        else:
            return f(mstar)

    def set_vmax(self, x):
        dmin3 = self.cosmo.comoving_distance(0.65).value ** 3
        self.vmax = np.array([4 * np.pi / 3 * self.footprint / self.sky * (
                self.cosmo.comoving_distance(self.zmax_lim(m_i)).value ** 3 - dmin3) for m_i in x])

    def get_vmax(self, x):
        dmin3 = self.cosmo.comoving_distance(0.65).value ** 3
        return np.array([4 * np.pi / 3 * self.footprint / self.sky * (
                self.cosmo.comoving_distance(self.zmax_lim(m_i)).value ** 3 - dmin3) for m_i in x])

    def get_number_galaxies(self, bin=False, zmin=None, zmax=None):
        if self.z.shape[0] == self.x.shape[0] and bin==False:
            return self.z.shape[0]
        if bin:
            z, x = self.select_galaxies(zmin, zmax)
            return z.shape[0]
