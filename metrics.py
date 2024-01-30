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

def bfs(df, fit_header, start_idx, visited, neutral_net):
    q = deque([start_idx])
    visited.add(start_idx)
    neutral_net.add(start_idx)

    while q:
        curr_arch_idx = q.popleft()
        curr_fit = df.at[curr_arch_idx, fit_header]
        nbrs = util.nbrs(df, curr_arch_idx)
        for nbr_idx in nbrs.index:
            if nbr_idx not in visited and df.at[nbr_idx, fit_header] == curr_fit:
                visited.add(nbr_idx)
                neutral_net.add(nbr_idx)
                q.append(nbr_idx)

def neutral_nets(df, fit_header):
    visited = set()
    nets = []
    for i in tqdm(range(len(df))):
        if i not in visited:
            net = set()
            bfs(df, image_dataset, fit_header, visited, net)
            if len(net) > 1:
                nets.append(net)
    return nets    

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
        for nbr_idx in nbrs.index:
            nbr_fit = df.at[nbr_idx, fit_header]
            # if the neighbor is fitter than the current architecture, the current architecture is not a local maximum
            if nbr_fit > curr_arch_fit:
                local_max = False
                break
        if local_max:
            count += 1
    return count