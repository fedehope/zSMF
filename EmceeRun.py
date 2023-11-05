import emcee
import numpy as np
from FileEmcee import FileEmcee
from Fileh5 import Fileh5


class EmceeRun(Fileh5):
    def __init__(self, file_emcee_obj):
        super().__init__(file_emcee_obj.folder, file_emcee_obj.info_file)
        self.best_params = None
        self.flat_samples = None
        self.file_emcee_obj = file_emcee_obj
        self.emcee_file = self.get_file()

        self.reader = emcee.backends.HDFBackend(self.emcee_file)
        self.samples = self.reader.get_chain()

        self.labels4 = [r'$a_{0}$', r'$a_{1}$', r'$a_{2}$', r'$a_{3}$']
        self.labels2 = [r'$\log(M_{*})$', r'$\alpha_{1}$']
        self.labels3 = [r'$\log(M_{*})$', r'$\alpha_{1}$', r'$\alpha_{2}$']
        self.labels6 = [r'$a_{0}$', r'$a_{1}$', r'$a_{2}$', r'$a_{3}$', r'$a_{4}$', r'$a_{5}$', r'$a_{6}$']
        self.labels8 = [r'$a_{0}$', r'$a_{1}$', r'$a_{2}$', r'$\alpha_{1}$', r'$\alpha_{2}$',
                        r'$a_{7}$', r'$a_{8}$', r'$a_{9}$']
        self.labels10 = [r'$a_{0}$', r'$a_{1}$', r'$a_{2}$', r'$a_{3}$', r'$a_{4}$', r'$a_{5}$', r'$a_{6}$',
                         r'$a_{7}$', r'$a_{8}$', r'$a_{9}$']

    def set_best_params(self, discard):
        self.flat_samples = self.reader.get_chain(discard=discard, thin=15, flat=True)
        mcmc = np.array(
            [np.percentile(self.flat_samples[:, i], [16, 50, 84]) for i in range(self.flat_samples.shape[1])])
        self.best_params = mcmc[:, 1]

    def get_best_params(self):
        return self.best_params

    @staticmethod
    def run_comparison(discard=150):
        emcee_file_02 = Fileh5(folder='emcee_runs', info_file='0.0_0.2')
        emcee_file_03 = Fileh5(folder='emcee_runs', info_file='0.0_0.3')
        emcee_file_04 = Fileh5(folder='emcee_runs', info_file='0.0_0.4')

        emcee_run_02 = EmceeRun(emcee_file_02)
        emcee_run_03 = EmceeRun(emcee_file_03)
        emcee_run_04 = EmceeRun(emcee_file_04)

        emcee_run_02.set_best_params(discard=discard)
        emcee_run_03.set_best_params(discard=discard)
        emcee_run_04.set_best_params(discard=discard)

        return np.array([emcee_run_02, emcee_run_03, emcee_run_04])
