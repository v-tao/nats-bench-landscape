import numpy as np
import util
from tqdm import tqdm

def global_max(df, image_dataset, epochs):
    # gets the fittest architecture for a given image dataset and number of epochs trained
    global_max_i = df[f"{image_dataset}TestAccuracy{epochs}Epochs"].idxmax()
    return df.loc[global_max_i]

def dists_to_arch(df, arch):
    # gets the distances of each architecture to a given architecture string
    dists = np.zeros(15625)
    for i in range(15625):
        dists[i] = util.edit_distance(df.loc[i]["ArchitectureString"], arch)
    return dists

def FDC(df, image_dataset, epochs):
    opt = global_max(df, image_dataset, epochs)
    fits = df[f"{image_dataset}TestAccuracy{epochs}Epochs"].values
    dists = dists_to_arch(df, opt["ArchitectureString"])

    # covariacne between fitnesses and distances
    cov_fd = np.cov(fits, dists)[0, 1]

    # variances of fitness and distances
    var_f = np.var(fits)
    var_d = np.var(dists)

    # Fitness Distance Correlation
    return (cov_fd) / np.sqrt(var_f * var_d)

def num_local_maxima(df, image_dataset, epochs):
    fit_col_name = f"{image_dataset}TestAccuracy{epochs}Epochs"
    fits = df[fit_col_name].values
    count = 0
    # iterate through all architectures
    for i in tqdm(range(len(df))):
        local_max = True
        curr_arch = df.loc[i]
        curr_arch_fit = curr_arch[fit_col_name]
        nbrs = util.nbrs(curr_arch["ArchitectureString"])
        # for each neighbor, check if fitness is less than current architecture
        for nbr_str in nbrs:
            nbr = df[df["ArchitectureString"] == nbr_str]
            nbr_fit = nbr[fit_col_name].values
            # if the neighbor is fitter than the current architecture, the current architecture is not a local maximum
            if nbr_fit > curr_arch_fit:
                local_max = False
                break
        if local_max:
            count += 1
    return count