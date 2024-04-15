# Setup Instructions
1. Follow the [NATS-Bench setup instructions](https://github.com/D-X-Y/NATS-Bench).
2. Install the requirements in `requirements.txt`.

# Overview of Important Files
`nats_bench_fitness_data.ipynb` contains the code used to generate `nats_bench.csv`, which pulls some of the NATS-Bench API queries into a csv format.
`main.ipynb` contains the code used to generate and analyze the data.
The folders CIFAR10, CIFAR100, and ImageNet contain the data used in the paper.
