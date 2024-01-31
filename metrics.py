import numpy as np
import util
from tqdm import tqdm
from collections import deque

def global_max(df, fit_header):
    # gets the fittest architecture for a given fitness metric
    global_max_i = df[fit_header].idxmax()
    return df.loc[global_max_i]

def dists_to_arch(df, arch):
    # gets the distances of each architecture to a given architecture string
    dists = np.zeros(len(df))
    for i in range(len(df)):
        dists[i] = util.edit_distance(df.loc[i]["ArchitectureString"], arch)
    return dists

def FDC(df, fit_header):
    opt = global_max(df, fit_header)
    fits = df[fit_header].values
    dists = dists_to_arch(df, opt["ArchitectureString"])

    # covariance between fitnesses and distances
    cov_fd = np.cov(fits, dists)[0, 1]

    # variances of fitness and distances
    var_f = np.var(fits)
    var_d = np.var(dists)

    # Fitness Distance Correlation
    return (cov_fd) / np.sqrt(var_f * var_d)

def bfs(df, fit_header, start_i, visited, neutral_net):
    q = deque([start_i])
    visited.add(start_i)
    neutral_net.add(start_i)

    while q:
        curr_arch_i = q.popleft()
        curr_fit = df.at[curr_arch_i, fit_header]
        nbrs = util.nbrs(df, curr_arch_i)
        for nbr_i in nbrs.index:
            if nbr_i not in visited and df.at[nbr_i, fit_header] == curr_fit:
                visited.add(nbr_i)
                neutral_net.add(nbr_i)
                q.append(nbr_i)

def neutral_nets(df, fit_header):
    visited = set()
    nets = []
    for i in tqdm(range(len(df))):
        if i not in visited:
            net = set()
            bfs(df, fit_header, i, visited, net)
            if len(net) > 1:
                nets.append(net)
    return nets    

def percolation_index(df, fit_header, net):
    # gets the number of different fitness values surrounding the neutral area
    values = set()
    for arch_i in net:
        for nbr_i in util.nbrs(df, arch_i).index:
            values.add(df.at[nbr_i, fit_header])
    return len(values)


def neutral_nets_analysis(df, fit_header, neutral_nets):
    nets_info = []
    for net in tqdm(neutral_nets):
        # get max and average distances between architectures
        net_list = list(net)
        arch_strs = df.loc[net_list, "ArchitectureString"].tolist()
        fit = df.at[net_list[0], fit_header]
        dists = []
        for i in range(len(arch_strs)):
            for j in range(i + 1, len(arch_strs)):
                dists.append(util.edit_distance(arch_strs[i], arch_strs[j]))  
        max_dist = max(dists)
        avg_dist = sum(dists)/len(dists)      

        net_info = {
            "Size": len(net),
            "Fitness": fit,
            "PercolationIndex": percolation_index(df, fit_header, net),
            "MaxEditDistance": max_dist,
            "AvgEditDistance": avg_dist,
        }
        nets_info.append(net_info)
    return nets_info

def num_local_maxima(df, fit_header):
    fits = df[fit_header].values
    count = 0
    # iterate through all architectures
    for i in tqdm(range(len(df))):
        local_max = True
        curr_arch = df.loc[i]
        curr_arch_fit = curr_arch[fit_header]
        nbrs = util.nbrs(df, i)
        # for each neighbor, check if fitness is less than current architecture
        for nbr_i in nbrs.index:
            nbr_fit = df.at[nbr_i, fit_header]
            # if the neighbor is fitter than the current architecture, the current architecture is not a local maximum
            if nbr_fit > curr_arch_fit:
                local_max = False
                break
        if local_max:
            count += 1
    return count