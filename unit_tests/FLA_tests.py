import unittest
from itertools import product
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from modules import util
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis
from config.Edge import Edge

class TestFLA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFLA, self).__init__(*args, **kwargs)
        df = pd.read_csv("../nats_bench.csv")
        EDGES = [Edge.CONV_1X1, Edge.CONV_3X3]

        # get all possible architectures using only the edges in EDGES
        # much smaller search space of 64 architectures can be analyzed by hand
        archs = [list(arch_str) for arch_str in product(EDGES, repeat=6)]
        self._genotypes = [util.edges2str(arch_str) for arch_str in archs]
        self._fits = df[df["ArchitectureString"].isin(self._genotypes)]["Cifar10TestAccuracy12Epochs"].values
        self._fits_dict = dict(zip(self._genotypes, self._fits))

        self._FLA = FitnessLandscapeAnalysis(self._fits, self._genotypes)
        self._global_max_i = np.argmax(self._fits)
        self._size = len(self._fits)

    def test_FDC(self):
        FDC = self._FLA.FDC()
        FDC_test = 0
        dists = np.zeros(self._size)
        # create distances array
        for i in range(self._size):
            dists[i] = util.edit_distance(self._genotypes[i], self._genotypes[self._global_max_i])
        self.assertEqual(dists[self._global_max_i], 0)
        fits_mean = np.mean(self._fits)
        dists_mean = np.mean(dists)
        fits_std = np.std(self._fits)
        dists_std = np.std(dists)

        for i in range(self._size):
            FDC_test += ((1/self._size) * (self._fits[i] - fits_mean) * (dists[i] - dists_mean))/(fits_std * dists_std)
        self.assertAlmostEqual(FDC_test, FDC)

if __name__ == "__main__":
    unittest.main()