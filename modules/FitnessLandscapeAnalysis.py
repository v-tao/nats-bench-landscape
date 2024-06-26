import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import random
import os
import csv
import json
from scipy import stats
from matplotlib import pyplot as plt
from modules import util
from config.Edge import Edge

class FitnessLandscapeAnalysis:
    """
    This class contains methods for calculating various metrics of the search space for the purposes of fitness landscape analysis

    Attributes:
        _fits (np.ndarray): array of fitnesses
        _global_max (int): index of fittest solution candidate
        _genotypes (list of strings): list of genotypes (in our case, architecture strings)
        _pheontypes (list of strings): list of phenotypes (in our case, unique architecture strings)
        _file_path (String): location where data will be saved
        _size (int): number of solution candidates
        _edges (set of Strings): set of edges to choose from
    
    Methods:
        get_fits(): returns the fitnesses of the fitness landscape
        get_dists_to_global_max(): calculates and returns the distance of each architecture to the global optimum
        get_global_max(): returns the index of the global optimum
        collect_data(): saves data for local maxima, weak and strong basins, autocorrelation run history, and neutral networks
        run_analysis(): runs general analysis of fitness landscape
        correlations(): returns the fitness vs. distance  correlations of the search space with the global maximum as the reference point
        neutral_net_bfs(start_i): uses BFS to obtain the neutral network around the given starting architecture
        neutral_nets(): returns a list of neutral networks
        percolation_index(net): returns the percolation index (number of unique fitness values surrounding the neutral network) of a neutral network
        neutral_nets_analysis(): runs a more in-depth analysis of the neutral networks
        local_maxima(): returns the indices of the local maxima
        random_walk(start_i, walk_len=100): generates a random walk along the landscape using one-edge adjustments
        random_walks(trials=200, walk_len=100, save=True): does many random walks
        weak_basin(start_i): returns the weak basin (architectures who have a strictly increasing path to the target architecture) around the given target architecture
        weak_basins(maxima, save=True): returns all the weak basins around all local maxima
        strong_baisns(weak_basins_dict): returns all the strong basins (architectures who have a strictly increasing path uniquely to one target architecture)
    """

    def __init__(self, fits, genotypes, phenotypes, file_path, edges={Edge.NONE, Edge.CONV_1X1, Edge.CONV_3X3, Edge.SKIP_CONNECT, Edge.AVG_POOL_3X3}):
        """
        Initialize a new instance of FitnessLandscapeAnalysis

        Parameters:
            fits (numpy.ndarray): array of fitnesses
            genotypes (list of strings): list of genotypes (in our case, architecture strings)
            phenotypes (list of strings): list of phenotypes (in our case, unique architecture strings)
            file_path (String): location where data will be saved
            edges (set of Strings): set of edges to choose from
        """
        self._fits = fits
        self._global_max = np.argmax(self._fits)
        self._genotypes = genotypes
        self._phenotypes = phenotypes
        self._file_path = file_path
        self._size = len(self._fits)
        self._edges = edges
        
    def get_fits(self):
        """
        Gets the fitnesses of the fitness landscape

        Parameters:
            none
        Returns:
            (numpy.ndarray): array of fitnesses
        """
        return self._fits
    
    def get_dists_to_global_max(self):
        """
        Calculates and returns the distance of each architecture to the global optimum

        Parameters:
            none
        Returns:
            (numpy.ndarray): array of distances to the global optimum
        """
        return util.dists_to_arch(self._genotypes, self._global_max)

    def get_global_max(self):
        """
        Returns the index of the global optimum

        Parameters:
            none
        Returns:
            (int) the index of the global optimum
        """
        return self._global_max

    def collect_data(self):
        """
        Saves data for local maxima, weak and strong basins, random walks, and neutral networks
        Parameters:
            none
        Returns:
            none
        """
        os.makedirs(f"{self._file_path}/data", exist_ok=True)
        maxima = self.local_maxima(save=True)
        weak_basins = self.weak_basins(maxima, save=True)
        self.strong_basins(weak_basins, save=True)
        self.neutral_nets(save=True)
        self.random_walks(save=True)

    def run_analysis(self):
        """
        Runs a fitness landscape analysis of the fitnesses and genotypes, and returns the corresponding object
        Parameters:
            None

        Returns:
            (object) object containing metrics analysis
        """
        # ========== DENSITY OF STATES ==========
        max_fitness = self._fits[self._global_max]
        fitness_diffs_from_max = np.array(self._fits) - max_fitness
        avg_fitness_diff_from_max = np.average(fitness_diffs_from_max)

        # ========== CORRELATIONS ==========
        corrs = self.correlations()

        # ========== LOCAL MAXIMA ==========
        with open(f"{self._file_path}/data/local_maxima.csv") as local_max_f:
            local_maxima = [int(i) for i in list(next(csv.reader(local_max_f)))]
        maxima_genotypes = [self._genotypes[i] for i in local_maxima]
        maxima_dists = [util.edit_distance(genotype, self._genotypes[self._global_max]) for genotype in maxima_genotypes]
        maxima_fits = [self._fits[i] for i in local_maxima]

        # ========== BASINS OF ATTRACTION ==========
        # ---------- WEAK BASINS ----------
        weak_basins = dict()
        weak_basin_sizes = []
        in_weak_basin = set()
        global_max_weak_basin_size = None
        for i in local_maxima:
            with open(f"{self._file_path}/data/weak_basins/local_max_{i}_weak_basin.csv") as weak_basin_f:
                weak_basin = list(next(csv.reader(weak_basin_f)))
                weak_basins[i] = weak_basin
                weak_basin_sizes.append(len(weak_basin))
                in_weak_basin.update(weak_basin)
                if i == self._global_max:
                    global_max_weak_basin_size = len(weak_basin)

        # ---------- STRONG BASINS ----------
        strong_basins = dict()
        strong_basin_sizes = []
        in_strong_basin = set()
        global_max_strong_basin_size = None
        for strong_basin_f_name in os.listdir(f"{self._file_path}/data/strong_basins"):
            with open(f"{self._file_path}/data/strong_basins/{strong_basin_f_name}") as strong_basin_f:
                local_max = int(strong_basin_f_name[10: -17])
                strong_basin = list(next(csv.reader(strong_basin_f)))
                # only count the basins that are not empty
                if len(strong_basin) > 0:
                    # store strong basin in a dictionary corresponding to its optimum
                    strong_basins[local_max] = strong_basin
                    # keep track of strong basin sizes
                    strong_basin_sizes.append(len(strong_basin))
                    # keep track of which architectures appear in a strong basin
                    in_strong_basin.update(strong_basin)
                    if local_max == self._global_max:
                        global_max_strong_basin_size = len(strong_basin)

        # ========== NEUTRALITY ==========
        neutral_nets = []

        # Open the CSV file and read its contents
        with open(f"{self._file_path}/data/neutral_networks.csv", newline='') as neutral_nets_f:
            for neutral_net_l in csv.reader(neutral_nets_f):
                # reader will read strings, we want ints
                neutral_net = [int(i) for i in neutral_net_l]
                neutral_nets.append(neutral_net)
    
        # get the fitness of each neutral net, since the neutral nets all have the same fitness can just take first architecture
        neutral_net_sizes = [len(net) for net in neutral_nets]

        # ---------- LARGEST NEUTRAL NETWORK ----------
        largest_neutral_net = neutral_nets[np.argmax(neutral_net_sizes)]
        largest_neutral_net_unique_phenotypes = set()
        largest_neutral_net_unique_nbr_genotypes = set()
        largest_neutral_net_unique_nbr_phenotypes = set()
        largest_neutral_net_unique_nbr_fits = set()
        largest_neutral_net_edit_dists = []
        for arch_i in largest_neutral_net:
            largest_neutral_net_unique_phenotypes.add(self._phenotypes[arch_i])
            # get all unique genotypes, phenotypes, and fitnesses of neighbors
            for nbr_i in util.nbrs(self._genotypes, arch_i):
                # do not add neighbors already in the neutral network
                if nbr_i not in largest_neutral_net:
                    largest_neutral_net_unique_nbr_genotypes.add(self._genotypes[nbr_i])
                    largest_neutral_net_unique_nbr_phenotypes.add(self._phenotypes[nbr_i])
                    largest_neutral_net_unique_nbr_fits.add(self._fits[nbr_i])
        # get all the edit distances between members of neutral network
        for i in range(len(largest_neutral_net)):
            for j in range(i + 1, len(largest_neutral_net)):
                arch_i_str = self._genotypes[i]
                arch_j_str = self._genotypes[j]
                largest_neutral_net_edit_dists.append(util.edit_distance(arch_i_str, arch_j_str))
        largest_neutral_net_max_edit_distance = max(largest_neutral_net_edit_dists)
        largest_neutral_net_avg_edit_distance = sum(largest_neutral_net_edit_dists) / len(largest_neutral_net_edit_dists)
        largest_neutral_net_fit = self._fits[largest_neutral_net[0]]
        
        # ========== RUGGEDNESS ==========
        random_walks = []
        autocorrs = dict()
        with open(f"{self._file_path}/data/200_random_length_100_walks.csv", newline='') as random_walks_f:
            for random_walk in csv.reader(random_walks_f):
                random_walks.append(random_walk)
        for lag in range(1, 21):
            autocorrs_specific_lag = []
            for random_walk in random_walks:
                walk_fits = [self._fits[int(i)] for i in random_walk]
                autocorr_specific_walk = stats.pearsonr(walk_fits[:-lag], walk_fits[lag:])[0]
                autocorrs_specific_lag.append(autocorr_specific_walk)
            autocorrs[lag] = sum(autocorrs_specific_lag)/len(autocorrs_specific_lag)
        

        summary = {
            "maxFitness": max_fitness,
            "avgFitnessDiffFromMax": avg_fitness_diff_from_max,
            "FDC": corrs["FDC"],
            "spearmanr": corrs["spearmanr"],
            "kendalltau": corrs["kendalltau"],
            "numLocalMaxima": len(local_maxima),
            "modality": len(local_maxima)/self._size,
            "localMaximaPearsonr": stats.pearsonr(maxima_dists, maxima_fits),
            "localMaximaSpearmanr": stats.spearmanr(maxima_dists, maxima_fits),
            "localMaximaKendallTau": stats.kendalltau(maxima_dists, maxima_fits),
            "numWeakBasins": len(weak_basins),
            "avgWeakBasinSize": sum(weak_basin_sizes)/len(weak_basins),
            "weakBasinExtent": len(in_weak_basin)/self._size,
            "fitnessWeakBasinSizePearsonr": stats.pearsonr(maxima_fits, weak_basin_sizes),
            "globalMaxWeakBasinExtent": global_max_weak_basin_size/self._size,
            "numStrongBasins": len(strong_basins),
            "avgStrongBasinSize": sum(strong_basin_sizes)/len(strong_basins),
            "strongBasinExtent": len(in_strong_basin)/self._size,
            "globalMaxStrongBasinExtent": global_max_strong_basin_size/self._size,
            "numNeutralNets": len(neutral_nets),
            "avgNeutralNetSize": sum(neutral_net_sizes)/len(neutral_nets),
            "maxNeutralNetSize": max(neutral_net_sizes),
            "largestNeutralNetFitness": largest_neutral_net_fit,
            "largestNeutralNetUniquePhenotypes": len(largest_neutral_net_unique_phenotypes),
            "largestNeutralNetUniqueNeighborGenotypes": len(largest_neutral_net_unique_nbr_genotypes),
            "largestNeutralNetUniqueNeighborPhenotypes": len(largest_neutral_net_unique_nbr_phenotypes),
            "largestNeutralNetMaxEditDistance": largest_neutral_net_max_edit_distance,
            "largestNeutralNetAvgEditDistance": largest_neutral_net_avg_edit_distance,
            "largestNeutralNetPercolationIndex": len(largest_neutral_net_unique_nbr_fits),
            "correlationLength": -1/np.log(autocorrs[1])
        }

        with open(f"{self._file_path}/summary.json", "w") as summary_f:
            json.dump(summary, summary_f)

        with open(f"{self._file_path}/autocorrelations.json", "w") as autocorrs_f:
            json.dump(autocorrs, autocorrs_f)
    
    # def generate_visualizations(self):
    #     os.makedirs(f"{self._file_path}/vis", exist_ok=True)
    #     # ========== FITNESS ==========
    #     plt.figure()
    #     plt.hist(self._fits, bins=100)
    #     plt.xlim(left=0, right=100)
    #     plt.ylim(bottom=0, top=3500)

    #     # Labels
    #     plt.xlabel("Test Accuracy")
    #     plt.ylabel("Number of Architectures")

    #     plt.savefig(f"{self._file_path}/vis/fitnesses.png")

    #     # ========== FITNESS/DISTANCE CORRELATION ==========
    #     dists = util.dists_to_arch(self._genotypes, self._global_max)
    #     plt.figure()
    #     plt.scatter(dists, self._fits, edgecolor="black", alpha=0.05)

    #     # Labels
    #     plt.xlabel("Distance to Global Maximum")
    #     plt.ylabel("Fitness")
        
    #     plt.savefig(f"{self._file_path}/vis/fits_dists.png")

    #     # ========== FITNESS/DISTANCE CORRELATION OPTIMA ONLY ==========
    #     with open(f"{self._file_path}/data/local_maxima.csv") as local_max_f:
    #         local_maxima = [int(i) for i in list(next(csv.reader(local_max_f)))]
    #     plt.figure()
    #     maxima_genotypes = [self._genotypes[i] for i in local_maxima]
    #     maxima_dists = [util.edit_distance(genotype, self._genotypes[self._global_max]) for genotype in maxima_genotypes]
    #     maxima_fits = [self._fits[i] for i in local_maxima]

    #     plt.scatter(maxima_dists, maxima_fits)
        
    #     # Labels
    #     plt.xlabel("Distance to Global Maximum")
    #     plt.ylabel("Fitness")

    #     plt.savefig(f"{self._file_path}/vis/optima_fits_dists.png")

    #     # =========== AUTOCORRELATION ==========
    #     # Extract lags and corresponding autocorrelations
    #     with open(f"{self._file_path}/autocorrelations.json", "r") as autocorrs_f:
    #         autocorrs_dict = json.load(autocorrs_f)
    #     lags = autocorrs_dict.keys()
    #     autocorrs = autocorrs_dict.values()

    #     plt.figure()
    #     plt.bar(lags, autocorrs)
    #     plt.ylim(bottom=-0.1, top=0.7)
    #     # Threshold between "difficult" and "straightforward"
    #     plt.axhline(y=0.15, color='gray', linestyle='--', alpha=0.5)

    #     # Labels
    #     plt.xlabel("Lag")
    #     plt.ylabel("Autocorrelation")

    #     plt.savefig(f"{self._file_path}/vis/autocorrelations.png")

    def correlations(self):
        """
        Returns the fitness vs. distance  correlations of the search space with the global maximum as the reference point

        Parameters:
            none

        Returns:
            (dict): dictionary of different correlation values
        """
        # distances are to the fittest architecture
        dists = util.dists_to_arch(self._genotypes, self._global_max)
        FDC = stats.pearsonr(self._fits, dists) # same as Pearson's correlation
        spearman = stats.spearmanr(self._fits, dists)
        kendall = stats.kendalltau(self._fits, dists)
        return {
            "FDC": FDC,
            "spearmanr": spearman,
            "kendalltau": kendall,
        }

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
            with open(f"{self._file_path}/data/neutral_networks.csv", "w", newline="") as f:
                csv_writer = csv.writer(f)
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
            save (boolean, deafult True): determines whether or not to save the local max data
        
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
                    # if the neighbor is at least as fit as the current architecture,
                    # the current architecture cannot be a local max
                    if self._fits[nbr_i] >= self._fits[i]:
                        local_max = False
                    # if the neighbor is at most as fit as the current architecture,
                    # the neighbor cannot be a local max
                    elif self._fits[nbr_i] <= self._fits[i]:
                        visited.add(nbr_i)
                # if all the neighbors are less fit, then the current architecture is a local maximum
                if local_max:
                    maxima.append(i)
        if save:
            with open(f"{self._file_path}/data/local_maxima.csv", "w", newline="") as f:
                csv.writer(f).writerow(maxima)
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
    
    def random_walks(self, trials=200, walk_len=100, save=True):
        """
        Performs many random walks

        Parameters:
            trials (int, default 200): number of random walks to take
            walk_len (int, default 100): walk length
            save (boolean, default True): determines whether or not to save the random walk data
        """
        walks = np.empty((200, 100), dtype=int)
        for i in tqdm(range(trials)):
            start_i = random.randint(0, self._size-1)
            walk = self.random_walk(start_i, walk_len)
            walks[i] = walk
        if save:
            np.savetxt(f"{self._file_path}/data/{trials}_random_length_{walk_len}_walks.csv", walks, delimiter=",", fmt="%d")
        return walks


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

    def weak_basins(self, maxima, save=True):
        """
        Returns all the weak basins of the search space, that is the weak basins of all optima

        Parameters:
            maxima (list of ints): list of local maxima
            save (boolean, deafult True): determines whether or not to save the autocorrelation walk data
        
        Returns:
            (dict): dictionary of weak basins where the key is the index of a local max and the value is the corresponding weak basin
        """
        basins = dict()
        for max_i in tqdm(maxima):
            basin = self.weak_basin(max_i)
            basins[max_i] = basin
            if save:
                os.makedirs(f"{self._file_path}/data/weak_basins", exist_ok=True)
                with open(f"{self._file_path}/data/weak_basins/local_max_{max_i}_weak_basin.csv", "w", newline="") as f:
                    csv.writer(f).writerow(list(basin))
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
                os.makedirs(f"{self._file_path}/data/strong_basins", exist_ok=True)
                with open(f"{self._file_path}/data/strong_basins/local_max_{opt}_strong_basin.csv", "w", newline="") as f:
                    csv.writer(f).writerow(list(strong_basin))
        return strong_basins_dict