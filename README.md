# FedTree

Small federated learning prototype using a tree-class subset of CIFAR-100.  
It supports several aggregation strategies:

**FedAvg**: standard weight averaging.
**FedProx**: FedAvg with a proximal term during local optimization.
**FedNova**: update normalization by local steps.

## Training

```bash
python3 train.py --dataset cifar100 --agg fedavg --rounds 30 --num_clients 4 \
    --local_epochs 5 --output_dir ./outputs/fedavg_example
```
Trains using FedAvg for 30 communication rounds with 4 clients.

```bash
python3 train.py --dataset cifar100 --agg fedprox --mu 0.01 --rounds 50 \
    --num_clients 8 --local_epochs 3 --output_dir ./outputs/fedprox_example
```
Uses FedProx with a proximal coefficient of 0.01.


```bash
python3 train.py --dataset cifar100 --agg fednova --rounds 40 --num_clients 6 \
    --local_epochs 2 --output_dir ./outputs/fednova_example
```
Applies FedNova where each client performs the same number of local epochs.

## Evaluation

After training you can evaluate and visualize results:

```bash
python3 evaluate.py --experiment_dir ./outputs/fedavg_example --plot
```
This computes loss, accuracy and class-wise metrics for the federated and centralized models and saves confusion matrices when `--plot` is set.

## Full arguments list

| Argument           | Default                  | Description                                                        |
| ------------------ | ------------------------ | ------------------------------------------------------------------ |
| `--dataset`        | `cifar100`               | Dataset to use (currently only `cifar100` is supported)            |
| `--model`          | `mobilenet_v3_small`     | Backbone model architecture                                        |
| `--agg`            | `fedavg`                 | Aggregation strategy: `fedavg`, `fedavgm`, `fedprox`, or `fednova` |
| `--mu`             | `0.0`                    | Proximal term coefficient for FedProx                              |
| `--rounds`         | `30`                     | Number of communication rounds                                     |
| `--local_epochs`   | `5`                      | Number of local training epochs per round                          |
| `--central_epochs` | `1`                      | Number of epochs for centralized baseline training                 |
| `--num_clients`    | `4`                      | Total number of clients in the federation                          |
| `--batch_size`     | `32`                     | Training batch size                                                |
| `--alpha`          | `0.5`                    | Dirichlet concentration for non-IID data split                     |
| `--lr`             | `1e-3`                   | Learning rate                                                      |
| `--seed`           | `42`                     | Random seed for reproducibility                                    |
| `--output_dir`     | `./outputs/experiment_1` | Path to save models, logs, and metadata                            |

