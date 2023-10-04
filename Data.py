import astropy.table as aTable
import numpy as np
import os

class Data:
    def __init__(self, file):
        self.data_dir = '/Users/federico/OneDrive - University College London/PhD/PhD_project/bgs_psmf/data'
        self.file = file
        self.data_file = os.path.join(self.data_dir, self.file)

    def get_data(self):
        if self.file.endswith('.hdf5'):
            return aTable.Table.read(self.data_file)
        elif self.file.endswith('.dat'):
            return np.loadtxt(self.data_file, unpack=True, usecols=[0, 1, 2])
