import unittest
from copy import deepcopy
import pandas as pd
import sys
sys.path.append("..")
from modules import util

EDGES = ["none", "nor_conv_1x1", "nor_conv_3x3", "skip_connect", "avg_pool_3x3"]
df = pd.read_csv("../nats_bench.csv")
arch_strs = list(df["ArchitectureString"])
class TestList2Str(unittest.TestCase):
    def test_list2str1(self):
        s = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        l = [(('nor_conv_1x1', 0),),
            (('none', 0), ('none', 1)),
            (('none', 0), ('none', 1), ('skip_connect', 2))]
        self.assertTrue(util.lists2str(l) == s)
    
    def test_str2edges1(self):
        s = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        edges = ["nor_conv_1x1", "none", "none", "none", "none", "skip_connect"]
        self.assertTrue(util.str2edges(s) == edges)
    
    def test_edges2str1(self):
        s = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        self.assertTrue(util.edges2str(util.str2edges(s)) == s)

    def test_edit_distance1(self):
        s1 = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        s2 = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        self.assertTrue(util.edit_distance(s1, s2) == 0)
    
    def test_edit_distance2(self):
        s1 = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        s2 = "|none~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        self.assertTrue(util.edit_distance(s1, s2) == 1)

    def test_edit_distance3(self):
        s1 = "|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|"
        s2 = "|none~0|+|nor_conv_1x1~0|nor_conv_1x1~1|+|nor_conv_1x1~0|nor_conv_1x1~1|none~2|"
        self.assertTrue(util.edit_distance(s1, s2) == 6)
    
    def test_nbr_strings1(self):
        s = "|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|"
        s_edges = util.str2edges(s)
        nbrs = util.nbrs(arch_strs, 0)
        nbrs_test = set()
        edges = util.str2edges(s)
        # check each one-edge adjustment from the starting architecture
        for i in range(len(edges)):
            for new_edge in EDGES:
                if s_edges[i] != new_edge:
                    nbr = deepcopy(s_edges)
                    nbr[i] = new_edge
                    nbr_str = util.edges2str(nbr)
                    nbr_i = df[df["ArchitectureString"] == nbr_str].index[0]
                    nbrs_test.add(nbr_i)
        self.assertEqual(nbrs_test, set(nbrs))

if __name__ == "__main__":
    unittest.main()