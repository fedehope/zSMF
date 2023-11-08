from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM, Planck13
from Data import Data


class COSMOS(Data):
    def __init__(self, file):
        super().__init__(file)
        self.footprint = 1.38  # deg^2
        self.sky = 41253  # deg^2
        self.cosmo = FlatLambdaCDM(H0=67.8, Om0=0.307)

        cosmos_all = self.get_data()

        mask = (cosmos_all['FLAG_HJMCC']==0) & (cosmos_all['FLAG_COSMOS']==1) & (cosmos_all['FLAG_PETER']==0)

        self.z = cosmos_all['ZPDF'][mask].data
        self.x = cosmos_all['MASS_MED'][mask].data

    def select_galaxies(self, zmin, zmax):
        """Return z, x, x_median, w_spec, vmax for galaxies between (zmin,zmax)"""
        mask_zlim = (self.z > zmin) & (self.z < zmax)

        z = self.z[mask_zlim]
        x = self.x[mask_zlim]
        mask_mlim = (x > self.mass_completeness_limit(z))

        return z[mask_mlim], x[mask_mlim]

    def mass_completeness_limit(self, z):
        x_values = [0.175, 0.5, 0.8, 1.125, 1.525, 2.0, 2.5, 3.125, 3.75, 4.4]
        y_values = [8.1, 8.7, 9.1, 9.3, 9.7, 9.9, 10.0, 10.1, 10.1, 10.1]

        f = interp1d(x_values, y_values)
        return f(z)


    def zmax_lim(self, mstar):
        y_values = [8.1, 8.7, 9.1, 9.3, 9.7, 9.9, 10.0, 10.1, 10.1, 10.1]
        x_values = [0.175, 0.5, 0.8, 1.125, 1.525, 2.0, 2.5, 3.125, 3.75, 4.4]

        f = interp1d(y_values, x_values)
        if mstar > y_values[-1]:
            return x_values[-1]
        else:
            return f(mstar)

    def get_number_galaxies(self):
        if self.z.shape[0] == self.x.shape[0]:
            return self.z.shape[0]
