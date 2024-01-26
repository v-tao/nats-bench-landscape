import numpy as np
import pandas as pd
from copy import deepcopy
from config.Edge import Edge

def str2lists(arch_str):
        """Shows how to read the string-based architecture encoding.
        Args:
          arch_str: the input is a string indicates the architecture topology, such as
                        |nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|
        Returns:
          a list of tuple, contains multiple (op, input_node_index) pairs.
        [USAGE]
        It is the same as the `str2structure` func in AutoDL-Projects:
          `https://github.com/D-X-Y/AutoDL-Projects/blob/main/xautodl/models/cell_searchs/genotypes.py`
        Every arch_str has the following sequential structure to describe the computational graph of a cell of an architecture:
        |operation_on_node_0~0| + |operation_on_node_0~0 | operation_on_node_1~1 | + ... + |operation_on_node_0~0  | ... | operation_on_node_n~n |

        where n + 1 represents the total amount of nodes.

        Each of set of bars explain the set of calculations required to calculate the data that a certain node contains for future operations. When there are multiple operations within each set of bars (each divided by +), that indicates a sum of the results of all those operations.
        ```
          arch = api.str2lists( '|nor_conv_1x1~0|+|none~0|none~1|+|none~0|none~1|skip_connect~2|' )
          print ('there are {:} nodes in this arch'.format(len(arch)+1)) # arch is a list
          for i, node in enumerate(arch):
            print('the {:}-th node is the sum of these {:} nodes with op: {:}'.format(i+1, len(node), node))
        ```
        """
        node_strs = arch_str.split("+")
        genotypes = []
        for unused_i, node_str in enumerate(node_strs):
            inputs = list(
                filter(lambda x: x != "", node_str.split("|"))
            )  # pylint: disable=g-explicit-bool-comparison
            for xinput in inputs:
                assert len(xinput.split("~")) == 2, "invalid input length : {:}".format(
                    xinput
                )
            inputs = (xi.split("~") for xi in inputs)
            input_infos = list([op, int(idx)] for (op, idx) in inputs)
            genotypes.append(input_infos)
        return genotypes

def lists2str(l):
    """
    =========
    lists2str
    =========
    Converts architecture from list format to string format.

    INPUTS:
    l (list): list representation of architecture

    OUTPUTS:
    s (String): string representation of architecture
    """
    first_node = f"|{l[0][0][0]}~{l[0][0][1]}|"
    second_node = f"|{l[1][0][0]}~{l[1][0][1]}|{l[1][1][0]}~{l[1][1][1]}|"
    third_node = f"|{l[2][0][0]}~{l[2][0][1]}|{l[2][1][0]}~{l[2][1][1]}|{l[2][2][0]}~{l[2][2][1]}|"
    return f"{first_node}+{second_node}+{third_node}"

def str2edges(arch_str):
  """
  =========
  str2edges
  =========
  Returns a list of edges of the architecture, in the order that 
  they appear in the architecture string.

  INPUTS:
  arch_str: string representation of architecture

  OUTPUTS:
  edges (list): list of architecture edges
  """

  arch = str2lists(arch_str)
  edges = []
  # iterate through nodes
  for i in range(len(arch)):
    # iterate through the incoming edges
    for j in range(i+1):
      edges.append(arch[i][j][0])
  return edges

def edges2str(edges):
  """
  =========
  edges2str
  =========
  Returns the string representation of the architecture given a list of edges.

  INPUTS:
  edges (list): list of architecture edges

  OUTPUTS:
  arch_str (String): string representation of the architecture
  """
  first_node = f"|{edges[0]}~0|"
  second_node = f"|{edges[1]}~0|{edges[2]}~1|"
  third_node = f"|{edges[3]}~0|{edges[4]}~1|{edges[5]}~2|"
  return f"{first_node}+{second_node}+{third_node}"

def edit_distance(arch1_str, arch2_str):
  """
  =============
  edit_distance
  =============
  Returns the edit distance (hamming distance) between two architectures.
  
  INPUTS:
  arch1_str (str): the string representation of the first architecture
  arch2_str (str): the string representation of the second architecture

  OUTPUTS:
  edit_distance (int): the edit distance between the two given architectures
  """
  arch1_edges = str2edges(arch1_str)
  arch2_edges = str2edges(arch2_str)
  assert(len(arch1_edges) == len(arch2_edges))
  edit_distance = 0
  for i in range(len(arch1_edges)):
    if arch1_edges[i] != arch2_edges[i]:
      edit_distance += 1
  return edit_distance

def nbrs(arch_str):
    """
    ===========
    nbrs
    ===========
    Returns list of architectures that are one edge changed from the input architecture.

    INPUTS:
    arch_str (String): string for architectures we want the neighbors of

    OUTPUTS:
    nbrs (set of Strings): set of strings representations of architectures
        that are one edge different from the input architecture (including edge removal)
    """
    arch = str2lists(arch_str)
    edges = {Edge.NONE, Edge.CONV_1X1, Edge.CONV_3X3, Edge.SKIP_CONNECT, Edge.AVG_POOL_3X3}
    nbrs = set()
    # iterate through each non-input node
    for i in range(len(arch)):
        # iterate through the incoming edges
        for j in range(i+1):
            # iterate through each edge possibility
            for edge in edges:
                # only adds the new architecture to the neighborhood if it is different
                # from the old architecture
                if arch[i][j][0] != edge:
                    nbr = deepcopy(arch)
                    nbr[i][j][0] = edge
                    nbrs.add(lists2str(nbr))
    return nbrs