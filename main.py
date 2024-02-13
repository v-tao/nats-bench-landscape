import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis

df = pd.read_csv("nats_bench.csv")
FLA = FitnessLandscapeAnalysis(df["CIFAR10TestAccuracy200Epochs"].values, list(df["ArchitectureString"].values))

autocorrs = []
for i in tqdm(range(10, 210, 10)):
    autocorrs.append(FLA.autocorrelation(trials=150, walk_len=i))

plt.xlabel("Walk Length (150 Trial)")
plt.ylabel("Autocorrelation")
plt.plot(range(10, 210, 10), autocorrs)
plt.savefig("autocorrs_walk_len.png")