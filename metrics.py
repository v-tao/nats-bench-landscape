import numpy as np
import util
from tqdm import tqdm
from collections import deque
import random

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

    cov_matrix = np.cov(fits, dists)

    # Fitness Distance Correlation
    return cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

def neutral_net_bfs(df, fit_header, start_i, visited, neutral_net):
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
            neutral_net_bfs(df, fit_header, i, visited, net)
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

def local_maxima(df, fit_header):
    # returns local_maxima
    fits = df[fit_header].values
    visited = set()
    maxima = []
    # iterate through all architectures
    for i in tqdm(range(len(df))):
        if i not in visited:
            local_max = True
            nbrs = util.nbrs(df, i)
            nbrs_i = nbrs.index.tolist()
            visited.add(i)
            # for each neighbor, check if fitness is less than current architecture
            for nbr_i in nbrs_i:
                # if the neighbor is greater, then the current arch cannot be a local maximum
                if fits[nbr_i] > fits[i]:
                    local_max = False
                # if the neighbor is smaller, then the neighbor cannot be a local maximum
                elif fits[nbr_i] < fits[i]:
                    visited.add(nbr_i)
            if local_max:
                maxima.append(i)
    return maxima


def num_local_maxima(df, fit_header):
    return len(local_maxima(df, fit_header))

def random_walk(df, fit_header, start_i, walk_len=100):
    # start a random walk at the given starting architecture for the given walk length
    curr_arch_i = start_i
    walk = [curr_arch_i]
    for i in range(walk_len - 1):
        # choose random neighbor index
        rand_nbr_i = random.choice(util.nbrs(df, curr_arch_i).index.tolist())
        walk.append(rand_nbr_i)
        curr_arch_i = rand_nbr_i
    return walk

def autocorrelation(df, fit_header, lag=1, trials=200, walk_len=100):
    # estimates the autocorrelation of a population for a certain lag
    autocorrs = np.zeros(trials)
    for i in tqdm(range(trials)):
        start_i = random.randint(0, len(df)-1)
        walk = random_walk(df, fit_header, start_i, walk_len)
        fits = df.loc[walk, fit_header].values
        cov_matrix = np.cov(fits[:-lag], fits[lag:])
        autocorr = cov_matrix[0, 1]/np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
        autocorrs[i] = (autocorr)
    print(np.average(autocorr))
    print(1/np.average(autocorr))
    return np.average(autocorr)

def weak_basin(df, fit_header, start_i):
    q = deque([start_i])
    visited = {start_i}
    basin = {start_i}

    while q:
        curr_arch_i = q.popleft()
        curr_fit = df.at[curr_arch_i, fit_header]
        nbrs = util.nbrs(df, curr_arch_i)
        for nbr_i in nbrs.index:
            if nbr_i not in visited and df.at[nbr_i, fit_header] < curr_fit:
                visited.add(nbr_i)
                basin.add(nbr_i)
                q.append(nbr_i)
    
    return basin



