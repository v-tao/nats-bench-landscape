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
        bit_encodings = []
        for arch in archs:
            bit_encoding = ""
            for edge in arch:
                if edge == Edge.CONV_1X1:
                    bit_encoding += "0"
                else:
                    bit_encoding += "1"
            bit_encodings.append(bit_encoding)
        self._genotypes = [util.edges2str(arch_str) for arch_str in archs]
        self._fits = df[df["ArchitectureString"].isin(self._genotypes)]["Cifar10TestAccuracy12Epochs"].values

        self._FLA = FitnessLandscapeAnalysis(self._fits, self._genotypes, edges=EDGES)
        self._global_max_i = np.argmax(self._fits)
        self._size = len(self._fits)
        keys = np.arange(self._size)
        vals = list(zip((self._fits), bit_encodings))
        self._fits_dict = dict(zip(keys, vals))

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

if __name__ == "__main__":
    
    unittest.main()