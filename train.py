import argparse
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from datasets import TreeSubsetDataset, get_transform
from models import get_model
from client import train_local
from server import Server

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def split_clients(class_indices, num_clients, alpha):
    client_idxs = [[] for _ in range(num_clients)]
    for c, idxs in class_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for i in range(num_clients):
            client_idxs[i].extend(splits[i].tolist())
    return client_idxs

def train_centralized(model, dataloader, device, lr, epochs):
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for _ in range(epochs):
        total_loss = 0
        total_samples = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
        losses.append(total_loss / total_samples)

    return model.cpu(), losses

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_transform(model_type=args.model)
    train_ds = TreeSubsetDataset(args.dataset, root="./data", train=True,
                                 transform=transform, class_names=None)
    test_ds = TreeSubsetDataset(args.dataset, root="./data", train=False,
                                transform=transform, class_names=train_ds.class_names)

    num_classes = len(train_ds.class_names)

    class_indices = train_ds.get_class_indices()
    client_idxs = split_clients(class_indices, args.num_clients, args.alpha)
    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=args.batch_size, shuffle=True)
        for idxs in client_idxs
    ]

    global_model = get_model(args.model, num_classes=num_classes, pretrained=True)
    server = Server(global_model, aggregation=args.agg, device=device)

    fed_avg_losses = []
    fed_client_losses = [[] for _ in range(args.num_clients)]

    for rnd in range(1, args.rounds + 1):
        client_states = []
        client_sizes = []
        round_losses = []

        global_state = None
        if args.agg == 'fedprox':
            global_state = server.get_model().state_dict()

        for cid, loader in enumerate(client_loaders):
            local_model = server.get_model()
            state_dict, num_samples, loss = train_local(
                local_model, loader, device,
                lr=args.lr, epochs=args.local_epochs,
                global_state=global_state, mu=args.mu if args.agg == 'fedprox' else 0.0
            )
            client_states.append(state_dict)
            client_sizes.append(num_samples)
            fed_client_losses[cid].append(loss)
            round_losses.append(loss)

        avg_loss = sum(sz * l for sz, l in zip(client_sizes, round_losses)) / sum(client_sizes)
        fed_avg_losses.append(avg_loss)
        print(f"Using aggregation: {args.agg}, round {rnd}, average loss: {avg_loss:.4f}")
        if args.agg == 'fednova':
            server.aggregate(client_states, client_sizes, client_steps=[args.local_epochs]*len(client_states))
        else:
            server.aggregate(client_states, client_sizes)
    
    torch.save(server.get_model().state_dict(), os.path.join(args.output_dir, "fed_model.pt"))
    np.save(os.path.join(args.output_dir, "fed_losses.npy"), np.array(fed_avg_losses))
    np.save(os.path.join(args.output_dir, "per_client_losses.npy"), np.array(fed_client_losses, dtype=object))

    centralized_model = get_model(args.model, num_classes=num_classes, pretrained=True)
    centralized_model, central_losses = train_centralized(
        centralized_model,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        device,
        lr=args.lr,
        epochs=args.central_epochs
    )

    torch.save(centralized_model.state_dict(), os.path.join(args.output_dir, "central_model.pt"))
    np.save(os.path.join(args.output_dir, "central_losses.npy"), np.array(central_losses))

    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "aggregation": args.agg,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "central_epochs": args.central_epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "seed": args.seed,
        "classes": train_ds.class_names
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--model', default='mobilenet_v3_small')
    parser.add_argument('--agg', default='fedavg', choices=['fedavg', 'fedavgm', 'fedprox', 'fednova'])
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--central_epochs', type=int, default=1)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='./outputs/experiment_1')

    args = parser.parse_args()
    main(args)
