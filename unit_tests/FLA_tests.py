import unittest
from itertools import product
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from modules import util
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis
from config.Edge import Edge
import csv

class TestFLA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFLA, self).__init__(*args, **kwargs)
        df = pd.read_csv("../nats_bench.csv")
        EDGES = {Edge.CONV_1X1, Edge.CONV_3X3}

        # get all possible architectures using only the edges in EDGES
        # much smaller search space of 64 architectures can be analyzed by hand
        archs = [list(arch_str) for arch_str in product(EDGES, repeat=6)]
        # this is so I can see the edit distances more easily
        self._bit_encodings = []
        for arch in archs:
            bit_encoding = ""
            for edge in arch:
                if edge == Edge.CONV_1X1:
                    bit_encoding += "0"
                else:
                    bit_encoding += "1"
            self._bit_encodings.append(bit_encoding)
        self._genotypes = [util.edges2str(arch_str) for arch_str in archs]
        self._fits = df[df["ArchitectureString"].isin(self._genotypes)]["Cifar10TestAccuracy12Epochs"].values

        self._FLA = FitnessLandscapeAnalysis(self._fits, self._genotypes, edges=EDGES)
        self._global_max_i = np.argmax(self._fits)
        self._size = len(self._fits)

    def test_FDC(self):
        FDC = self._FLA.FDC()
        self.assertAlmostEqual(FDC, -0.1738316237, places=2)

    def test_local_maxima(self):        
        self.assertEqual(set(self._FLA.local_maxima()), {12, 32, 35, 45, 49, 54})
    
    def test_neutral_nets(self):
        self.assertEqual(self._FLA.neutral_nets(), [{3,7}])

    def test_percolation_index(self):
        net = self._FLA.neutral_nets()[0]
        self.assertEqual(self._FLA.percolation_index(net), 9)
    
    def test_neutral_nets_analysis(self):
        net_info = self._FLA.neutral_nets_analysis()[0]
        self.assertEqual(net_info["Size"], 2)
        self.assertEqual(net_info["Fitness"], 84.53)
        self.assertEqual(net_info["MaxEditDistance"], 1)
        self.assertEqual(net_info["MaxEditDistance"], 1)
    
    def test_weak_basin(self):
        self.assertEqual(self._FLA.weak_basin(12), {13, 14, 4, 28, 44, 8, 40, 24, 0, 10, 9, 15, 30, 6, 29, 60, 46, 41, 56, 26, 1, 2, 11, 25, 21, 38, 27, 33, 48, 3, 5, 18, 50, 7, 23, 22, 31, 55, 20, 63})
    
    def test_strong_basins(self):
        self.assertEqual(self._FLA.strong_basins(self._FLA.weak_basins()), {
            12: {8, 9},
            32: {36},
            35: {39, 42, 43},
            45: set(),
            49: set(),
            54: set()
        })

if __name__ == "__main__":
    unittest.main()