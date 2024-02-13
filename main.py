import pandas as pd
import numpy as np
import random
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis

df = pd.read_csv("nats_bench.csv")
search_spaces = ["CIFAR100", "ImageNet"]
genotypes = list(df["ArchitectureString"].values)
for space in search_spaces:
    fits = df[f"{space}TestAccuracy200Epochs"]
    file_path = f"data/{space}"
    FLA = FitnessLandscapeAnalysis(fits, genotypes, file_path)
    FLA.collect_data()