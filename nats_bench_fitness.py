from nats_bench import create
import pandas as pd
from tqdm import tqdm

api = create(None, 'tss', fast_mode=True, verbose=False)

data = []
for i in tqdm(range(15625)):
    data.append({
        "CIFAR10TestAccuracy200Epochs": api.get_more_info(i, "cifar10", hp=200, is_random=False)['test-accuracy'],
        "CIFAR100TestAccuracy200Epochs": api.get_more_info(i, "cifar100", hp=200, is_random=False)['test-accuracy'],
        "ImageNetTestAccuracy200Epochs": api.get_more_info(i, "ImageNet16-120", hp=200, is_random=False)['test-accuracy'],
        "ArchitectureString": api.arch(i),
    })
df = pd.DataFrame(data)
df.to_csv("nats_bench.csv", index=False)