import emcee
import numpy as np
from FileEmcee import FileEmcee


class EmceeRun(FileEmcee):
    def __init__(self, file_emcee_obj):
        super().__init__(file_emcee_obj.z_dep, file_emcee_obj.bin_test, file_emcee_obj.info_file)
        self.best_params = None
        self.flat_samples = None
        self.file_emcee_obj = file_emcee_obj

        self.emcee_file = self.get_file()

        self.reader = emcee.backends.HDFBackend(self.emcee_file)
        self.samples = self.reader.get_chain()

        self.labels4 = [r'$a_{0}$', r'$a_{1}$', r'$a_{3}$', r'$a_{3}$']
        self.labels2 = [r'$\log(M_{*})$', r'$\alpha_{1}$']

    def set_best_params(self, discard):
        self.flat_samples = self.reader.get_chain(discard=discard, thin=15, flat=True)
        mcmc = np.array(
            [np.percentile(self.flat_samples[:, i], [16, 50, 84]) for i in range(self.flat_samples.shape[1])])
        self.best_params = mcmc[:, 1]

    def get_best_params(self):
        return self.best_params