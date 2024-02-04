import numpy as np
import util
from tqdm import tqdm
from collections import deque
import random
class FitnessLandscapeAnalysis:
    def __init__(self, fits, genotypes):
        self._fits = fits
        self._genotypes = genotypes
        self._size = len(self._fits)

    def FDC():
        """
        Returns the fitness distance correlation (FDC) of the search space with the global maximum as the reference point

        Parameters:
            none

        Returns:
            (float): FDC of the search space
        """
        # distances are to the fittest architecture
        dists = util.dists_to_arch(self._genotypes, np.argmax(self._fits))
        return np.corrcoef(self._fits, dists)[0, 1]

    def neutral_net_bfs(start_i):
        """
        Returns the neutral network around a given starting architecture

        Parameters:
            start_i (int): index of starting architecture

        Returns:
            (set of ints): indices corresponding to a neutral network around the starting architecture
        """
        q = deque([start_i])
        visited = {start_i}
        net = {start_i}
        while q:
            curr_i = q.popleft()
            nbrs = util.nbrs(self._genotypes, curr_i)
            for nbr_i in nbrs:
                # only explore neighbors that have the same fitness as the current architecture
                if nbr_i not in visited and self._fits[nbr_i] == self._fits[curr_i]:
                    visited.add(nbr_i)
                    net.add(nbr_i)
                    q.append(nbr_i)
        return net

    def neutral_nets():
        """
        Returns the neutral networks of a search space

        Parameters:
            none

        Returns:
            (list of set of ints): list of neutral networks of the search space
        """
        nets = []
        # do bfs starting from each architecture to search for neutral networks
        for i in tqdm(range(len(self._fits))):
            net = self.neutral_net_bfs(self._fits, self._genotypes, start_i)
            if len(net) > 1:
                nets.append(net)
        return nets

    def percolation_index(net):
        """
        Returns the percolation index (number of unique neighboring fitness values) of a given neutral network

        Parameters:
            net (set of ints): indices corresponding to a neutral network
        
        Returns:
            (int): percolation index (number of unique neighboring fitness values)
        """
        # gets the number of different fitness values surrounding the neutral area
        values = set()
        # go through all neighbors of all architectures and record their fitness values
        for arch_i in net:
            for nbr_i in util.nbrs(self._genotypes, arch_i):
                values.add(self._fits[nbr_i])
        return len(values)

    def neutral_nets_analysis():
        """
        Returns analysis of each neutral network in a search space

        Parameters:
            none

        Returns:
            (list of dicts): analysis of each neutral network
        """
        nets = neutral_nets(self._fits, self._genotypes)
        nets_info = []
        # run analysis for each neutral net
        for net in tqdm(nets):
            # convert neutral net to a list so it can be indexed to find the fitness of the neutral net
            net_list = list(net)
            net_fit = self._fits[net[0]]
            net_strs = [self._genotypesarch_strs[arch_i] for arch_i in net_list]
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
                "PercolationIndex": percolation_index(neutral_net),
                "MaxEditDistance": max_dist,
                "AvgEditDistance": avg_dist,
            }
            nets_info.append(net_info)
        return nets_info

    def local_maxima():
        """
        Returns a list of indices corresponding to local maxima in the search space

        Parameters:
            none
        
        Returns:
            (list of ints): indices corresponding to local maxima in the search space
        """
        visited = set()
        maxima = []
        # iterate through all architectures
        for i in tqdm(range(self._size)):
            if i not in visited:
                # create a flag for if the architecture is a maximum or not
                local_max = True
                nbrs = util.nbrs(self._genotypes, i)
                visited.add(i)
                # for each neighbor, check if fitness is less than current architecture
                for nbr_i in nbrs:
                    # if the neighbor is greater, then the current arch cannot be a local maximum
                    if self._fits[nbr_i] > self._fits[i]:
                        local_max = False
                    # if the neighbor is smaller, then the neighbor cannot be a local maximum
                    elif self._fits[nbr_i] < self._fits[i]:
                        visited.add(nbr_i)
                # if all the neighbors are smaller, then the current architecture is a local maximum
                if local_max:
                    maxima.append(i)
        return maxima

    def random_walk(start_i, walk_len=100):
        """
        Performs a random walk on the search space, starting at a given architecture index.

        Parameters:
            start_i (int): index of starting architecture
            walk_len (int, default 100): walk length
        
        Returns:
            (list of ints): indices of a random walk in the search space
        """
        # start a random walk at the given starting architecture for the given walk length
        curr_i = start_i
        walk = [curr_i]
        for i in range(walk_len - 1):
            # choose random neighbor index
            rand_nbr_i = random.choice(util.nbrs(self._genotypes, curr_i))
            walk.append(rand_nbr_i)
            curr_i = rand_nbr_i
        return walk

    def autocorrelation(lag=1, trials=200, walk_len=100):
        """
        Estimates the autocorrelation for the population given a certain lag.

        Parameters:
            lag (int, default 1): lag used to compute autocorrelation
            trials (int, default 200): number of samples to take
            walk_len (int, default 100): walk length
        
        Returns:
            (float): estimate of autocorrelation
        """
        autocorrs = np.empty(trials)
        # get the autocorrelation for many random walks
        for i in tqdm(range(trials)):
            start_i= random.randint(0, self._size-1)
            walk = self.random_walk(start_i, walk_len)
            walk_fits = [self._fits[i] for i in walk]
            autocorrs[i] = np.corrcoef(walk_fits[:-lag], walk_fits[lag:])[0, 1]
        return np.average(autocorr)

    def weak_basin(start_i):
        """
        Returns the weak basin (set of architectures with a strictly increasing path) of a given architecture

        Parameters:
            start_i (int): index of starting architecture
        
        Returns:
            (set of ints): weak basin (set of architectures with a strictly increasing path)
        """
        q = deque([start_i])
        visited = {start_i}
        basin = {start_i}

        while q:
            curr_i = q.popleft()
            nbrs = util.nbrs(self._genotypes, curr_arch_i)
            for nbr_i in nbrs:
                # add neighbors who are worse than current architecture
                if nbr_i not in visited and self._fits[nbr_i] < self._fits[curr_arch_i]:
                    visited.add(nbr_i)
                    basin.add(nbr_i)
                    q.append(nbr_i)
        return basin

    def weak_basins():
        """
        Returns all the weak basins of the search space, that is the weak basins of all optima

        Parameters:
            none
        
        Returns:
            (dict): dictionary of weak basins where the key is the index of a local max and the value is the corresponding weak basin
        """
        maxima = self.local_maxima(df, fit_header)
        basins = dict()
        for maximum in tqdm(maxima):
            basins[maximum] = self.weak_basin(df, fit_header, maximum)
        return basins

    def strong_basins(weak_basins_dict):
        """
        Finds the strong basins corresponding to the weak basins. A strong basin contains only points that belong to one weak basin

        Parameters:
            none

        Returns:
            (dict): dictionary of strong basins where the key is the index of a local max and the value is the corresponding strong basin
        """
        # given a dictionary of weak basins of optima, find the strong basins of the corresponding optima
        basins = self.weak_basins_dict.values()
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