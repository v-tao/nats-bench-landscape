import numpy as np
import util
from tqdm import tqdm
from collections import deque
import random

def FDC(fits, arch_strs):
    """
    Returns the fitness distance correlation (FDC) of the search space with the global maximum as the reference point

    Parameters:
        fits (numpy.ndarray): fitness values corresponding to architecture indices
        arch_strs (list of Strings): architecture strings corresponding to architecture indices

    Returns:
        (float): FDC of the search space
    """
    dists = util.dists_to_arch(arch_strs, np.argmax(fits)) # np.argmax(fits) gets the index of the fittest architecture
    return np.corrcoef(fits, dists)[0, 1]

def neutral_net_bfs(fits, arch_strs, start_i):
    """
    Returns the neutral network around a given starting architecture

    Parameters:
        fits (numpy.ndarray): fitness values corresponding to architecture indices
        arch_strs (list of Strings): architecture strings corresponding to architecture indices
        start_i (int): index of starting architecture

    Returns:
        (set of ints): indices corresponding to a neutral network around the starting architecture
    """
    q = deque([start_i])
    visited = {start_i}
    net = {start_i}
    while q:
        curr_i = q.popleft()
        nbrs = util.nbrs(arch_strs, curr_arch_i)
        for nbr_i in nbrs.index:
            # only explore neighbors that have the same fitness as the current architecture
            if nbr_i not in visited and fits[nbr_i] == fits[curr_i]:
                visited.add(nbr_i)
                net.add(nbr_i)
                q.append(nbr_i)
    return net

def neutral_nets(fits, arch_strs):
    """
    Returns the neutral networks of a search space

    Parameters:
        fits (numpy.ndarray): fitness values corresponding to architecture indices
        arch_strs (list of Strings): architecture strings corresponding to architecture indices

    Returns:
        (list of set of ints): list of neutral networks of the search space
    """
    nets = []
    # do bfs starting from each architecture to search for neutral networks
    for i in tqdm(range(len(fits))):
        net = neutral_net_bfs(fits, arch_strs, start_i)
        if len(net) > 1:
            nets.append(net)
    return nets

def percolation_index(fits, arch_strs, net):
    """
    Returns the percolation index (number of unique neighboring fitness values) of a given neutral network

    Parameters:
        fits (numpy.ndarray): fitness values corresponding to architecture indices
        arch_strs (list of Strings): architecture strings corresponding to architecture indices
        net (set of ints): indices corresponding to a neutral network
    
    Returns:
        (int): percolation index (number of unique neighboring fitness values)
    """
    # gets the number of different fitness values surrounding the neutral area
    values = set()
    # go through all neighbors of all architectures and record their fitness values
    for arch_i in net:
        for nbr_i in util.nbrs(arch_strs, arch_i):
            values.add(fits[nbr_i])
    return len(values)


def neutral_nets_analysis(fits, arch_strs):
    """
    
    """
    nets = neutral_nets(fits, arch_strs)
    nets_info = []
    # run analysis for each neutral net
    for net in tqdm(nets):
        # convert neutral net to a list so it can be indexed to find the fitness of the neutral net
        net_list = list(net)
        net_fit = fits[net[0]]
        net_strs = [arch_strs[arch_i] for arch_i in net_list]
        dists = []

        # calculate edit distance between all pairs of architectures in the neutral net
        for i in range(len(net_strs)):
            for j in range(i + 1, len(net_strs)):
                dists.append(util.edit_distance(net_strs[i], net_strs[j]))  
        max_dist = max(dists)
        avg_dist = sum(dists)/len(dists)      

        net_info = {
            "Size": len(neutral_net),
            "Fitness": neutral_net_fit,
            "PercolationIndex": percolation_index(fits, arch_strs, neutral_net),
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
    fits = df[fit_header].values
    # get the weak basin for a particular architecture
    q = deque([start_i])
    visited = {start_i}
    basin = {start_i}

    while q:
        curr_arch_i = q.popleft()
        nbrs_i = util.nbrs(df, curr_arch_i).index.tolist()
        for nbr_i in nbrs_i:
            if nbr_i not in visited and fits[nbr_i] < fits[curr_arch_i]:
                visited.add(nbr_i)
                basin.add(nbr_i)
                q.append(nbr_i)
    return basin

def weak_basins(df, fit_header):
    maxima = local_maxima(df, fit_header)
    basins = dict()
    for maximum in tqdm(maxima):
        basins[maximum] = weak_basin(df, fit_header, maximum)
    return basins

def strong_basins(weak_basins_dict):
    # given a dictionary of weak basins of optima, find the strong basins of the corresponding optima
    basins = weak_basins_dict.values()
    not_unique = set()
    # get all of the archs that appear in more than one weak basin
    for basin1 in basins:
        for basin2 in basins:
            if basin1 != basin2:
                not_unique.update(basin1 & basin2)
    strong_basins_dict = dict()
    for k in weak_basins_dict.keys():
        strong_basins_dict[k] = weak_basins_dict[k] - not_unique
    return strong_basins_dict