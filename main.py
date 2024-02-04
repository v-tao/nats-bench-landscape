import pandas as pd
import numpy as np
import random
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis

df = pd.read_csv("nats_bench.csv")
FLA = FitnessLandscapeAnalysis(df["Cifar10TestAccuracy12Epochs"].values, list(df["ArchitectureString"].values))

# data = FLA.run_analysis()

# df2 = pd.DataFrame([data])
# df2.to_csv("fla_test.csv", index=False)