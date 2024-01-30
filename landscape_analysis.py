import argparse
import pandas as pd
import numpy as np
import util
import metrics

FITNESS_DATA = pd.read_csv("nats_bench.csv")
IMAGE_DATASETS = ["Cifar10", "Cifar100", "ImageNet16"]
EPOCHS = [12, 200]

data = []
for image_dataset in IMAGE_DATASETS:
    for epoch in EPOCHS:
        fit_header = f"{image_dataset}TestAccuracy{epoch}Epochs"
        FDC = metrics.FDC(FITNESS_DATA, fit_header)
        num_local_maxima = metrics.num_local_maxima(FITNESS_DATA, fit_header)
        modality = num_local_maxima/len(FITNESS_DATA)
        row = {
            "ImageDataset": image_dataset,
            "Epochs": epoch,
            "FDC": FDC,
            "LocalMaxima": num_local_maxima,
            "Modality": modality
        }
        data.append(row)

df = pd.DataFrame(data)
df.to_csv("fla_test.csv", index=False)