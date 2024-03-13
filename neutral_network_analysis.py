import csv
import numpy as np
import pandas as pd
from nats_bench import create
import modules.util as util

api = create(None, 'tss', fast_mode=True, verbose=False)
genotypes = list(pd.read_csv("nats_bench.csv")["ArchitectureString"].values)

neutral_net_genotypes = []
neutral_net_nbr_genotypes = []
largest_nn_sizes = []

for ss_name in ["CIFAR10", "CIFAR100", "ImageNet"]:
    with open(f"{ss_name}/data/neutral_networks.csv", "r") as nn_f:
        # get the sizes of the neutral networks
        r = csv.reader(nn_f)
        neutral_nets = [list(nn_l) for nn_l in r]
        neutral_net_sizes = np.array([len(neutral_net) for neutral_net in neutral_nets])
        largest_neutral_net = neutral_nets[np.argmax(neutral_net_sizes)]
        largest_nn_sizes.append(len(largest_neutral_net))
        nn_genotypes = set()
        nn_nbr_genotypes = set()
        for arch in largest_neutral_net:
            nn_genotypes.add(api.get_unique_str(arch))
            nbrs = util.nbrs(genotypes, int(arch))
            for nbr in nbrs:
                nn_nbr_genotypes.add(api.get_unique_str(nbr))
        neutral_net_genotypes.append(neutral_net_genotypes)
        neutral_net_nbr_genotypes.append(nn_nbr_genotypes)
print("CIFAR10 Neighbors")
print(largest_nn_sizes[0])
print(len(neutral_net_nbr_genotypes[0]))

print("CIFAR100 Neighbors")
print(largest_nn_sizes[1])
print(len(neutral_net_nbr_genotypes[1]))

print("ImageNet Neighbors")
print(largest_nn_sizes[2])
print(len(neutral_net_nbr_genotypes[2]))