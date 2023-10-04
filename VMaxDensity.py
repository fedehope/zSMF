import numpy as np

from BGS import BGS
from astropy.cosmology import Planck13

class VmaxDensity(BGS):
    def __int__(self, bgs_obj):
        BGS.__init__(bgs_obj.file, bgs_obj.is_bgs_bright)

    def histogram_norm(self, zmin, zmax):
        z, x, x_median, w_spec, vmax = self.select_galaxies(zmin, zmax)
        weights = (w_spec / vmax)
        hist, bin_edges = np.histogram(x_median, bins=40, range=(6., 13.), weights=weights)

        v_zmin = Planck13.comoving_volume(zmin).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3
        v_zmax = Planck13.comoving_volume(zmax).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3

        mask = (self.z > zmin) & (self.z < zmax)
        _w = self.w_spec / (self.vmax.clip(v_zmin, v_zmax) - v_zmin)
        _h, _b = np.histogram(self.x_median[mask], bins=40, range=(6., 13.), weights=_w[mask])
        return hist, bin_edges, _h, _b