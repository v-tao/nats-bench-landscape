import numpy as np
import util
from tqdm import tqdm
from collections import deque

def format_name(image_dataset, epochs):
    # gets the column name for the corresponding image dataset and number of epochs
    return f"{image_dataset}TestAccuracy{epochs}Epochs"

def global_max(df, image_dataset, epochs):
    # gets the fittest architecture for a given image dataset and number of epochs trained
    global_max_i = df[f"{image_dataset}TestAccuracy{epochs}Epochs"].idxmax()
    return df.loc[global_max_i]

def dists_to_arch(df, arch):
    # gets the distances of each architecture to a given architecture string
    dists = np.zeros(len(df))
    for i in range(len(df)):
        dists[i] = util.edit_distance(df.loc[i]["ArchitectureString"], arch)
    return dists

# def bfs(start, visited, neutral_net):
#     q = deque([start])
#     visited.add(start)
#     neutral_area.add(start)

#     while q:
#         current_arch = q.popleft()
#         nbr = util.nbrs()
#         for nbr in 

def neutrality(df, image_dataset, epochs):
    fit_header = format_name(image_dataset, epochs)
    neutral_nets = []
    visited = set()
    curr_arch = df.loc[i]
    curr_fit = curr_arch[fit_header]
    q = deque([0])

    while q:
        arch_i = q.popleft()
        if arch_i not in visited:
            visited.add(arch)
            nbrs = util.nbrs(curr_arch["ArchitectureString"])
            nbrs_i = nbrs.index.tolist()
            q.extend(nbr for nbr in nbrs_i)

    # for i in tqdm(range(len(df))):
    #     neutral_net = [i]
    #     curr_arch = df.loc[i]
    #     curr_fit = curr_arch[fit_header]
    #     nbrs = util.nbrs(curr_arch["ArchitectureString"])
    #     # check if any neighbors have the same fitness
    #     for nbr_str in nbrs:
    #         nbr = df[df["ArchitectureString"] == nbr_str]
    #         nbr_fit = nbr[fit_header].values
    #         # if the neighbr has the same fitness, add to neutral network
    #         if nbr_fit == curr_fit:
    #             neutral_net.append(nbr.index[0])
    #     # if the neutral network contains more than the original architecture, it is a neutral network
    #     if len(neutral_net) > 1:
    #         neutral_nets.add(tuple(neutral_net))
    # return neutral_nets


def FDC(df, image_dataset, epochs):
    fit_header = format_name(image_dataset, epochs)
    opt = global_max(df, image_dataset, epochs)
    fits = df[fit_header].values
    dists = dists_to_arch(df, opt["ArchitectureString"])

    # covariacne between fitnesses and distances
    cov_fd = np.cov(fits, dists)[0, 1]

    # variances of fitness and distances
    var_f = np.var(fits)
    var_d = np.var(dists)

    # Fitness Distance Correlation
    return (cov_fd) / np.sqrt(var_f * var_d)

def num_local_maxima(df, image_dataset, epochs):
    fit_header = format_name(image_dataset, epochs)
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