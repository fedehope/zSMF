import numpy as np
from scipy import integrate

from BGS import BGS


class ZSchechterModel(BGS):
    def __init__(self, bgs, zmin, zmax, z0):
        super().__init__(bgs.file)
        self.zmin = zmin
        self.zmax = zmax
        self.z0 = z0

        self.z, self.x, self.x_median, self.w_spec, self.vmax = self.select_galaxies(zmin, zmax)
        self.mlim = self.mass_completeness_limit(self.z)

    @staticmethod
    def phi(x, z, a0, a1, a2, a3):
        logM = a0 + a1 * z
        alpha1 = a2 + a3 * z

        term0 = np.exp(-10 ** (x - logM))
        term1 = 10 ** ((alpha1 + 1) * (x - logM))
        return term0 * term1

    @staticmethod
    def phi_vectorised(x, z, a0, a1, a2, a3):
        logM = a0 + a1 * z
        alpha1 = a2 + a3 * z

        term0 = np.log(10) * np.exp(-10 ** (x - logM[:, None]))
        term1 = 10 ** ((alpha1 + 1)[:, None] * (x - logM[:, None]))
        return term0 * term1

    def integral_phi(self, a0, a1, a2, a3, z0=False):
        if z0 is not True:
            return np.array([integrate.quad(ZSchechterModel.phi, self.mlim[i],
                                            13., args=(self.z[i], a0, a1, a2, a3))[0] for i in range(self.z.shape[0])])
        return np.array([integrate.quad(ZSchechterModel.phi, self.mlim[i],
                                        13., args=(self.z0, a0, a1, a2, a3))[0] for i in range(self.z.shape[0])])

    def log_likelihood(self, a0, a1, a2, a3):
        I = self.integral_phi(a0, a1, a2, a3)
        q = ZSchechterModel.phi_vectorised(self.x, self.z, a0, a1, a2, a3)
        a = np.log(np.sum(q, axis=1)) - np.log(I)
        return a * self.w_spec

    def log_prior(self, theta):
        a0, a1, a2, a3 = theta
        if 9.5 < a0 < 13.5 and \
                0. < a1 < 4. and \
                -4. < a2 < 0. and \
                0. < a3 < 3.:
            return 0
        return -np.inf

    def posterior(self, theta):
        a0, a1, a2, a3 = theta
        l = self.log_likelihood(a0, a1, a2, a3)
        return self.log_prior(theta) + np.sum(l)

    def normalisation(self, best_params):
        a0, a1, a2, a3 = best_params
        # v_zmin = Planck13.comoving_volume(self.zmin).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3
        # v_zmax = Planck13.comoving_volume(self.zmax).value * Planck13.h ** 3 * self.f_area  # (Mpc/h)^3
        m_max = 13.
        m_min = 6.
        nbin = 40
        bin_size = (m_max - m_min) / nbin
        I = integrate.quad(ZSchechterModel.phi, self.mlim.min(), 13., args=(self.z0, a0, a1, a2, a3))[0]
        return np.sum((self.w_spec * bin_size) / (self.vmax * I))