import numpy as np

from BGS import BGS
from astropy.cosmology import Planck13

from HST import HST


class VmaxDensityHst(HST):
    def __int__(self, hst_obj):
        HST.__init__(hst_obj.file)

    def histogram_norm(self, zmin, zmax, bins=40, range=(6., 13.)):
        z, x = self.select_galaxies(zmin, zmax)
        self.set_vmax(x)
        vmax = self.vmax
        weights = (1. / vmax)
        hist, bin_edges = np.histogram(x, bins=bins, range=range, weights=weights)

        v_zmax = 4 * np.pi / 3 * self.footprint / self.sky * (self.cosmo.comoving_distance(zmax).value ** 3)
        v_zmin = 4 * np.pi / 3 * self.footprint / self.sky * (self.cosmo.comoving_distance(zmin).value ** 3)

        # mask = (self.z > zmin) & (self.z < zmax)
        # _w = 1. / (self.get_vmax(self.x).clip(v_zmin, v_zmax) - v_zmin)
        # _h, _b = np.histogram(self.x[mask], bins=bins, range=(6., 13.), weights=_w[mask])
        return hist, bin_edges