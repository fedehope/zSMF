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
        self.samples = emcee_run.samples
        self.labels4 = emcee_run.labels4
        self.labels8 = emcee_run.labels8
        self.flat_samples = None
        self.z_lin = np.linspace(0, 0.65, 100)

    def plot_selected_data(self, zmin, zmax):
        z, x, x_median, w_spec, vmax = self.select_galaxies(zmin, zmax)

        plt.scatter(self.z, self.x_median, s=1, c='lightgrey', label='BGS Bright')
        plt.scatter(z, x_median, s=1, c='C0', label='BGS Bright Mlim')
        plt.plot(self.z_lin, self.mass_completeness_limit_bgs(self.z_lin), color='orange', linewidth=0.9)

        plt.legend(loc='lower right', fontsize=10, markerscale=10, handletextpad=0.1)
        plt.xlabel(r"redshift ($z$)")
        plt.xlim(0., 0.65)
        plt.ylabel(r"best-fit $\log M_*$")
        plt.ylim(6., 13.2)

    def plot_emcee_samples(self):
        ndim = len(self.labels4)

        fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels4[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")


    def plot_emcee_samples8(self):
        ndim = len(self.labels8)

        fig, axes = plt.subplots(ndim, figsize=(10, 10), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(self.samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(self.samples))
            ax.set_ylabel(self.labels8[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")

    @staticmethod
    def plot_emcee_corner(flat_samples, labels):
        fig = corner.corner(flat_samples, labels=labels, quantiles=(0.16, 0.50, 0.84), show_titles=True);
        # plt.show()

    @staticmethod
    def plot_vmax_hist(h, b, _h, _b, **plot_params):
        plt.step(b[:-1], h, where='post', linewidth=0.8, **plot_params)
        plt.step(_b[:-1], _h, where='post', color='k', linestyle='--', linewidth=0.8)
        plt.yscale('log')
        plt.ylim(1e-5, 4e-2)
        plt.xlim(6, 13)

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

        plt.xlabel(r'$\log M_*$', fontsize=15)
        plt.ylabel(r'$p(\log M_*)$ [$({\rm Mpc}/h)^{-3}{\rm dex}^{-1}$]', fontsize=15)

    @staticmethod
    # TODO to finish it
    def plot_zschechter_error(z0, flat_samples, **plot_params):
        x = np.linspace(6, 13, 100)
        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            a0, a1, a2, a3 = flat_samples[ind]
            zschechter = ZSchechterModel.phi(x, z0, a0, a1, a2, a3)
            norm = ZSchechterModel.normalisation(flat_samples[ind])
            plt.plot(x, norm * zschechter, **plot_params)

            plt.yscale('log')
            plt.ylim(1e-5, 4e-2)
            plt.xlim(7, 13)

            plt.xlabel(r'$\log M_*$', fontsize=15)
            plt.ylabel(r'$p(\log M_*)$ [$({\rm Mpc}/h)^{-3}{\rm dex}^{-1}$]', fontsize=15)
