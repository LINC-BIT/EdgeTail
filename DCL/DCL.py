
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.utils import resample


def get_class_counts(dataset, indices=None):
    if indices is None:
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.targets)[indices]
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = np.zeros(len(np.unique(dataset.targets)))
    class_counts[unique] = counts
    return class_counts

def update_weights(class_performance, class_counts, initial_weights):
    new_weights = initial_weights * (1 + class_performance)
    return new_weights / np.sum(new_weights) * len(class_counts)


def progressively_balanced_sampling(trainset, class_weights):
    labels = np.array(trainset.targets)
    indices = np.arange(len(labels))
    sample_weights = np.array([class_weights[label] for label in labels])
    #print(len(trainset))
    sampled_indices = np.random.choice(
        indices,
        size=len(trainset),
        replace=True,
        p=sample_weights / sample_weights.sum()
    )

    balanced_trainset = Subset(trainset, sampled_indices)
    #print(len(balanced_trainset))
    return balanced_trainset,sampled_indices