import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# 假设 trainset 是一个 PyTorch 数据集
# 例如：trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def square_root_biased_sampling(trainset, oversample_factor=1.0, undersample_factor=1.0):
    # 计算每个类别的样本数量
    class_counts = np.bincount(trainset.targets)
    #class_counts = [label for _, label in trainset.dataset]
    N_max = np.max(class_counts)
    N_min = np.min(class_counts)

    # 计算每个类别的目标样本数
    if undersample_factor==1.0:
       target_counts = (np.sqrt(N_max * class_counts) * oversample_factor).astype(int)
       target_counts = np.clip(target_counts, a_min=(class_counts * undersample_factor).astype(int), a_max=N_max).astype(int)

    if oversample_factor==1.0:
       target_counts = (np.sqrt(N_max * class_counts) * undersample_factor).astype(int)
       target_counts = np.clip(target_counts, a_min=N_min, a_max=N_max).astype(int)

    # 创建新的索引列表
    new_indices = []
    for class_idx in range(len(class_counts)):
        class_indices = np.where(np.array(trainset.targets) == class_idx)[0]
        np.random.shuffle(class_indices)

        if target_counts[class_idx] > len(class_indices):
            selected_indices = np.random.choice(class_indices, target_counts[class_idx], replace=True)
        else:
            selected_indices = class_indices[:target_counts[class_idx]]

        new_indices.extend(selected_indices)

    # 创建新的 DataLoader
    balanced_trainset = Subset(trainset, new_indices)
    return balanced_trainset

# 设置过采样因子和欠采样因子
