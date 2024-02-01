import argparse
import pandas as pd
import numpy as np
import random
import util
import metrics
import visualization as vis
import matplotlib.pyplot as plt

FITNESS_DATA = pd.read_csv("nats_bench.csv")
IMAGE_DATASETS = ["Cifar10", "Cifar100", "ImageNet16"]
EPOCHS = [12, 200]

fit_header = "Cifar10TestAccuracy12Epochs"
weak_basins = metrics.weak_basins(FITNESS_DATA, fit_header)
strong_basins = metrics.strong_basins(weak_basins)
print(strong_basins)

# data = []
# for image_dataset in IMAGE_DATASETS:
#     for epoch in EPOCHS:
#         fit_header = f"{image_dataset}TestAccuracy{epoch}Epochs"
#         nets = metrics.neutral_nets(FITNESS_DATA, fit_header)
#         nets_analysis = metrics.neutral_nets_analysis(FITNESS_DATA, fit_header, nets)
#         FDC = metrics.FDC(FITNESS_DATA, fit_header)
#         num_local_maxima = metrics.num_local_maxima(FITNESS_DATA, fit_header)
#         modality = num_local_maxima/len(FITNESS_DATA)
#         correlation_length = 1/metrics.autocorrelation(FITNESS_DATA, fit_header)
#         row = {
#             "ImageDataset": image_dataset,
#             "Epochs": epoch,
#             "FDC": FDC,
#             "LocalMaxima": num_local_maxima,
#             "Modality": modality,
#             "CorrelationLength": correlation_length
#         }
#         data.append(row)

# df = pd.DataFrame(data)
# df.to_csv("fla_test.csv", index=False)
# neutral_networks = pd.DataFrame(nets_analysis)
# neutral_networks.to_csv("neutral_networks.csv", index=False)