import corner
import numpy as np
from BGS import BGS
from EmceeRun import EmceeRun
from ZSchechterModel import ZSchechterModel

# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt

class Plotter(BGS, EmceeRun):
    def __init__(self, bgs, emcee_run):
        BGS.__init__(self, bgs.file)
        EmceeRun.__init__(self, emcee_run.file_emcee_obj)
        self.z_lin = np.linspace(0, 0.65, 100)

    def plot_selected_data(self, zmin, zmax):
        z, x, x_median, w_spec, vmax = self.select_galaxies(zmin, zmax)

        plt.scatter(self.z, self.x_median, s=1, c='lightgrey', label='BGS Bright')
        plt.scatter(z, x_median, s=1, c='C0', label='BGS Bright Mlim')
        plt.plot(self.z_lin, self.mass_completeness_limit(self.z_lin), color='orange', linewidth=0.9)

        plt.legend(loc='lower right', fontsize=10, markerscale=10, handletextpad=0.1)
        plt.xlabel(r"redshift ($z$)")
        plt.xlim(0., 0.65)
        plt.ylabel(r"best-fit $\log M_*$")
        plt.ylim(6., 13.2)

    def plot_emcee_samples(self):

        if self.z_dep:
            ndim = 4
            # labels = [r'$a_{0}$', r'$a_{1}$', r'$a_{3}$', r'$a_{3}$']

            # TODO fix the repetition
            fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)

            for i in range(ndim):
                ax = axes[i]
                ax.plot(self.samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(self.samples))
                ax.set_ylabel(self.labels4[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
        else:
            ndim = 2
            # labels = [r'$\log(M_{*})$', r'$\alpha_{1}$']

            fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)

            for i in range(ndim):
                ax = axes[i]
                ax.plot(self.samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(self.samples))
                ax.set_ylabel(self.labels2[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")

    # TODO maybe you can use self.flat_samples and self.labels
    def plot_emcee_corner(self, flat_samples, labels):
        fig = corner.corner(flat_samples, labels=labels, quantiles=(0.16, 0.50, 0.84), show_titles=True);
        plt.show()

    @staticmethod
    def plot_vmax_hist(h, b, _h, _b, **plot_params):
        plt.step(b[:-1], h, where='pre', linewidth=0.8, **plot_params)
        plt.step(_b[:-1], _h, where='pre', color='k', linestyle='--', linewidth=0.8)
        plt.yscale('log')
        plt.ylim(1e-5, 4e-2)
        plt.xlim(7, 13)

        plt.xlabel(r'$\log M_*$')
        plt.ylabel(r'$p(\log M_*)$ [$({\rm Mpc}/h)^{-3}{\rm dex}^{-1}$]')

    @staticmethod
    def plot_zschechter(x, z0, norm, best_params, **plot_params):
        a0, a1, a2, a3 = best_params
        zschechter = ZSchechterModel.phi(x, z0, a0, a1, a2, a3)
        plt.plot(x, norm * zschechter, **plot_params)

        plt.yscale('log')
        plt.ylim(1e-5, 4e-2)
        plt.xlim(7, 13)

        plt.xlabel(r'$\log M_*$')
        plt.ylabel(r'$p(\log M_*)$ [$({\rm Mpc}/h)^{-3}{\rm dex}^{-1}$]')