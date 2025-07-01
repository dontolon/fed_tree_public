import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
import os

TREE_CLASS_SUBSETS = {
    "cifar100": ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    "cifar10": [],  # CIFAR-10 doesn't include trees, but placeholder
    "tinyimagenet": []  # Placeholder for possible extension
}

def get_transform(model_type='mobilenet'):
    if model_type == 'mobilenet':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else: 
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class TreeSubsetDataset(Dataset):
    def __init__(self, dataset_name, root, train, transform, download=True, class_names=None):
        assert dataset_name in ["cifar100"], "Only CIFAR-100 supported for now."

        if dataset_name == "cifar100":
            base = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)

        self.base = base
        all_classes = base.classes

        if class_names is None:
            class_names = TREE_CLASS_SUBSETS[dataset_name]

        # Map class names to original indices
        name_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.original_class_indices = [name_to_idx[cls] for cls in class_names]
        self.label_map = {orig: new for new, orig in enumerate(self.original_class_indices)}

        # Filter dataset
        self.indices = [i for i, label in enumerate(base.targets) if label in self.original_class_indices]
        self.class_names = class_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, orig_label = self.base[real_idx]
        new_label = self.label_map[orig_label]
        return img, new_label

    def get_class_indices(self):
        """Returns dict: class_idx -> [dataset_indices]"""
        class_indices = defaultdict(list)
        for ds_idx, real_idx in enumerate(self.indices):
            orig = self.base.targets[real_idx]
            new = self.label_map[orig]
            class_indices[new].append(ds_idx)
        return class_indices
