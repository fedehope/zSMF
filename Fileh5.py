import os


class Fileh5:
    def __init__(self, folder, info_file):
        self.emcee_runs_dir = '/Users/federico/OneDrive - University College London/PhD/PhD_project/bgs_psmf/emcee_runs'
        self.working_dir = os.path.join(self.emcee_runs_dir, folder)
        self.info_file = info_file
        self.h5_file = self.get_file()

    def get_list_file(self):
        return [os.path.join(self.working_dir, file) for file in os.listdir(self.working_dir)]

    def get_file(self):
        list_file = self.get_list_file()

        found_file = [file for file in list_file if self.info_file in file]

        if len(found_file) > 1:
            indx = input(f'Multiple files with that <info_file> \n {list(enumerate(found_file))} \n'
                         f' - Tell me the index of the file you want to analyse: ')
            return found_file[int(indx)]
        else:
            return found_file[0]
