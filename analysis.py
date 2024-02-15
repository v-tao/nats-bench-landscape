import pandas as pd
import os
from modules.FitnessLandscapeAnalysis import FitnessLandscapeAnalysis

data_loc = "data" # location of data
analysis_loc = "analysis" # location of analysis
vis_loc = "visualization"

df = pd.read_csv("nats_bench.csv")
genotypes = list(df["ArchitectureString"].values)


for search_space in os.listdir(data_loc):
    FLA = FitnessLandscapeAnalysis(df[f"{search_space}TestAccuracy200Epochs"], genotypes, f"{data_loc}/{search_space}")
    # os.makedirs(f"{analysis_loc}/{search_space}", exist_ok=True)
    # FLA.run_analysis(f"{analysis_loc}/{search_space}")
    os.makedirs(f"{vis_loc}/{search_space}", exist_ok=True)
    FLA.generate_visualizations(f"{analysis_loc}/{search_space}", f"{vis_loc}/{search_space}")