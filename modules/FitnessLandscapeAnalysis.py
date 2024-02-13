import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import random
import os
import csv
from modules import util
from config.Edge import Edge

class FitnessLandscapeAnalysis:
    """
    This class contains methods for calculating various metrics of the search space for the purposes of fitness landscape analysis

    Attributes:
        _fits (np.ndarray): array of fitnesses
        _genotypes (list of strings): list of genotypes
        _file_path (String): location where data will be saved
        edges (set of Strings): set of edges to choose from
    
    Methods:
        run_analysis(): runs general analysis of fitness landscape
        FDC(): calculates the fitness distance correlation to the global maximum
        neutral_net_bfs(start_i): uses BFS to obtain the neutral network around the given starting architecture
        neutral_nets(): returns a list of neutral networks
        percolation_index(net): returns the percolation index (number of unique fitness values surrounding the neutral network) of a neutral network
        neutral_nets_analysis(): runs a more in-depth analysis of the neutral networks
        local_maxima(): returns the indices of the local maxima
        random_walk(start_i, walk_len=100): generates a random walk along the landscape using one-edge adjustments
        autocorrelation(lag=1, trials=200, walk_len=100): estimates the autocorrelation for the population
        weak_basin(start_i): returns the weak basin (architectures who have a strictly increasing path to the target architecture) around the given target architecture
        weak_basins(): returns all the weak basins around all local maxima
        strong_baisns(weak_basins_dict): returns all the strong basins (architectures who have a strictly increasing path uniquely to one target architecture)
    """

    def __init__(self, fits, genotypes, file_path, edges={Edge.NONE, Edge.CONV_1X1, Edge.CONV_3X3, Edge.SKIP_CONNECT, Edge.AVG_POOL_3X3}):
        """
        Initialize a new instance of FitnessLandscapeAnalysis

        Parameters:
            fits (numpy.ndarray): array of fitnesses
            genotypes (list of strings): list of genotypes
            file_path (String): location where data will be saved
            edges (set of Strings): set of edges to choose from
        """
        self._fits = fits
        self._genotypes = genotypes
        self._file_path = file_path
        self._size = len(self._fits)
        self._edges = edges

    def run_analysis(self):
        """
        Runs a fitness landscape analysis of the fitnesses and genotypes, and returns the corresponding object
        Parameters:
            none

        Returns:
            (object) object containing metrics analysis
        """
        FDC = self.FDC()
        local_maxima = self.local_maxima()
        autocorrelation = self.autocorrelation(trials=10, walk_len=10)
        weak_basins = self.weak_basins()
        strong_basins = self.strong_basins(weak_basins)
        summary = {
            "FDC": FDC,
            "NumLocalMaxima": len(local_maxima),
            "Modality": len(local_maxima)/self._size,
            "CorrelationLength": 1/autocorrelation,
            "NumWeakBasins": len(weak_basins),
            "NumStrongBasins": len(strong_basins)
        }
    def FDC(self):
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

    def neutral_net_bfs(self, start_i):
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
            nbrs = util.nbrs(self._genotypes, curr_i, edges=self._edges)
            for nbr_i in nbrs:
                # only explore neighbors that have the same fitness as the current architecture
                if nbr_i not in visited and self._fits[nbr_i] == self._fits[curr_i]:
                    visited.add(nbr_i)
                    net.add(nbr_i)
                    q.append(nbr_i)
        return net

    def neutral_nets(self, save=True):
        """
        Returns the neutral networks of a search space

        Parameters:
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data

        Returns:
            (list of set of ints): list of neutral networks of the search space
        """
        nets = []
        visited = set()
        # do bfs starting from each architecture to search for neutral networks
        for i in tqdm(range(len(self._fits))):
            # bfs should find an entire network at once, so no need to revisit a node that is already in a network
            if i not in visited:
                net = self.neutral_net_bfs(i)
                visited = visited | net
                if len(net) > 1:
                    nets.append(net)
        if save:
            with open(f"{self._file_path}/neutral_networks.csv", "w", newline="") as f:
                csv_writer = csv_writer(f)
                for net in nets:
                    csv_writer.writerow(net)
        return nets

    def percolation_index(self, net):
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
            for nbr_i in util.nbrs(self._genotypes, arch_i, edges=self._edges):
                # do not add the fitness of the neutral net
                if nbr_i not in net:
                    values.add(self._fits[nbr_i])
        return len(values)

    def neutral_nets_analysis(self):
        """
        Returns analysis of each neutral network in a search space

        Parameters:
            none

        Returns:
            (list of dicts): analysis of each neutral network
        """
        nets = self.neutral_nets()
        nets_info = []
        # run analysis for each neutral net
        for net in tqdm(nets):
            # convert neutral net to a list so it can be indexed to find the fitness of the neutral net
            net_list = list(net)
            net_fit = self._fits[net_list[0]]
            net_strs = [self._genotypes[arch_i] for arch_i in net_list]
            dists = []

            # calculate edit distance between all pairs of architectures in the neutral net
            for i in range(len(net_strs)):
                for j in range(i + 1, len(net_strs)):
                    dists.append(util.edit_distance(net_strs[i], net_strs[j]))  
            max_dist = max(dists)
            avg_dist = sum(dists)/len(dists)      

            net_info = {
                "Size": len(net),
                "Fitness": net_fit,
                "PercolationIndex": self.percolation_index(net),
                "MaxEditDistance": max_dist,
                "AvgEditDistance": avg_dist,
            }
            nets_info.append(net_info)
        return nets_info

    def local_maxima(self, save=True):
        """
        Returns a list of indices corresponding to local maxima in the search space

        Parameters:
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data
        
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
                nbrs = util.nbrs(self._genotypes, i, edges=self._edges)
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
        if save:
            np.savetxt(f"{np._file_name}/local_maxima.csv", maxima, delimiter=",")
        return maxima

    def random_walk(self, start_i, walk_len=100):
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
            rand_nbr_i = random.choice(util.nbrs(self._genotypes, curr_i, edges=self._edges))
            walk.append(rand_nbr_i)
            curr_i = rand_nbr_i
        return walk

    def autocorrelation(self, lag=1, trials=200, walk_len=100, save=True):
        """
        Estimates the autocorrelation for the population given a certain lag.

        Parameters:
            lag (int, default 1): lag used to compute autocorrelation
            trials (int, default 200): number of samples to take
            walk_len (int, default 100): walk length
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data
        
        Returns:
            (float): estimate of autocorrelation
        """
        autocorrs = np.empty(trials)
        history = np.empty((200, 100))
        # get the autocorrelation for many random walks
        for i in tqdm(range(trials)):
            start_i= random.randint(0, self._size-1)
            walk = self.random_walk(start_i, walk_len)
            walk_fits = [self._fits[i] for i in walk]
            autocorrs[i] = np.corrcoef(walk_fits[:-lag], walk_fits[lag:])[0, 1]
            history[i] = walk

        if save:
            # save the walk history for each autocorrelation, and save each autocorrelation
            os.makedirs(f"{self._file_path}/autocorrelation", exists_ok=True)
            np.savetxt(f"{self._file_path}/autocorrelation/walk_history_lag_{lag}_trials_{trials}_walk_len_{walk_len}.csv", history, delimiter=",")
            np.savetxt(f"{self._file_path}/autocorrelation/autocorrelations_lag_{lag}_trials_{trials}_walk_len_{walk_len}.csv", autocorrs, delimiter=",")
        
        return np.average(autocorrs)

    def weak_basin(self, start_i):
        """
        Returns the weak basin (set of architectures with a strictly increasing path) of a given architecture

        Parameters:
            start_i (int): index of starting architecture
        
        Returns:
            (set of ints): weak basin (set of architectures with a strictly increasing path)
        """
        q = deque([start_i])
        visited = {start_i}
        basin = set()

        while q:
            curr_i = q.popleft()
            nbrs = util.nbrs(self._genotypes, curr_i, edges=self._edges)
            for nbr_i in nbrs:
                # add neighbors who are no better than current architecture
                if nbr_i not in visited and self._fits[nbr_i] <= self._fits[curr_i]:
                    visited.add(nbr_i)
                    basin.add(nbr_i)
                    q.append(nbr_i)
        return basin

    def weak_basins(self, save=True):
        """
        Returns all the weak basins of the search space, that is the weak basins of all optima

        Parameters:
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data
        
        Returns:
            (dict): dictionary of weak basins where the key is the index of a local max and the value is the corresponding weak basin
        """
        maxima = self.local_maxima()
        basins = dict()
        for max_i in tqdm(maxima):
            basins[max_i] = self.weak_basin(max_i)
            if save:
                os.makedirs(f"{self._file_path}/weak_basins", exists_ok=True)
                np.savetxt(f"{self._file_path}/weak_basins/local_max_{i}_weak_basin", basins[max_i], delimiter=",")
        return basins

    def strong_basins(self, weak_basins_dict, save=True):
        """
        Finds the strong basins corresponding to the weak basins. A strong basin contains only points that belong to one weak basin

        Parameters:
            weak_basins_dict (dict): dictionary of weak basins
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data

        Returns:
            (dict): dictionary of strong basins where the key is the index of a local max and the value is the corresponding strong basin
        """
        # given a dictionary of weak basins of optima, find the strong basins of the corresponding optima
        basins = weak_basins_dict.values()
        not_unique = set()
        # get all of the archs that appear in more than one weak basin
        for basin1 in basins:
            for basin2 in basins:
                if basin1 != basin2:
                    not_unique.update(basin1 & basin2)
        strong_basins_dict = dict()
        for opt in weak_basins_dict.keys():
            strong_basin = weak_basins_dict[opt] - not_unique
            strong_basins_dict[opt] = strong_basin
            if save:
                os.makedirs(f"{self._file_path}/strong_basins", exists_ok=True)
                np.savetxt(f"{self._file_path}/strong_basins/local_max_{opt}_strong_basin", strong_basin, delimiter=",")
        return strong_basins_dict