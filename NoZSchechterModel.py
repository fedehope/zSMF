import numpy as np
from scipy import integrate

from BGS import BGS


class NoZSchechterModel(BGS):
    def __init__(self, bgs, zmin, zmax):
        super().__init__(bgs.file)
        self.zmin = zmin
        self.zmax = zmax

        self.z, self.x, self.x_median, self.w_spec, self.vmax = self.select_galaxies(zmin, zmax)
        self.mlim = self.mass_completeness_limit(self.z)

    @staticmethod
    def phi(x, logM, alpha1):
        term0 = np.log(10) * np.exp(-10 ** (x - logM))
        term1 = 10 ** ((alpha1 + 1) * (x - logM))
        return term0 * term1

    @staticmethod
    def phi_double(x, logM, alpha1, alpha2):
        term0 = np.log(10) * np.exp(-10 ** (x - logM))
        term1 = 10 ** ((alpha1 + 1) * (x - logM))
        term2 = 10 ** ((alpha2 + 1) * (x - logM))
        return term0 * (term1 + term2)

    def integral_phi(self, logM, alpha1):
        return np.array([integrate.quad(NoZSchechterModel.phi, self.mlim[i], 13., args=(logM, alpha1))[0] for i in
                         range(len(self.mlim))])

    def log_likelihood(self, logM, alpha1):
        I = self.integral_phi(logM, alpha1)
        q = NoZSchechterModel.phi(self.x, logM, alpha1)
        a = np.log(np.sum(q, axis=1)) - np.log(I)
        return a * self.w_spec

    def log_prior(self, theta):
        logM, alpha1 = theta
        if 9.5 < logM < 13.5 and \
                -5. < alpha1 < 0.:
            return 0
        return -np.inf

    def posterior(self, theta):
        logM, alpha1 = theta
        l = self.log_likelihood(logM, alpha1)
        return self.log_prior(theta) + np.sum(l)