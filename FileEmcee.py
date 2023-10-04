import os

class FileEmcee:
    def __init__(self, z_dependence, bin_test, info_file):
        self.emcee_runs_dir = '/Users/federico/OneDrive - University College London/PhD/PhD_project/bgs_psmf/emcee_runs'
        self.bin_tests_dir = '/Users/federico/OneDrive - University College London/PhD/PhD_project/bgs_psmf/emcee_runs/bin_tests_runs'
        self.gmm_dir = '/Users/federico/OneDrive - University College London/PhD/PhD_project/bgs_psmf/gmm'

        self.z_dep = z_dependence
        self.bin_test = bin_test
        self.info_file = info_file

    def get_list_file(self):
        if self.bin_test:
            if self.z_dep:
                return [os.path.join(self.bin_tests_dir, file) for file in os.listdir(self.bin_tests_dir) if
                        file.startswith('Z')]
            else:
                return [os.path.join(self.bin_tests_dir, file) for file in os.listdir(self.bin_tests_dir) if
                        file.startswith('Noz')]

    def get_file(self):
        list_file = self.get_list_file()

        found_file = [file for file in list_file if self.info_file in file]

        if len(found_file) > 1:
            return found_file
        else:
            return found_file[0]