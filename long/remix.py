import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter

class RemixDataset(Dataset):
    def __init__(self, dataset, cls_num,beta=2.0,tau=0.2, kappa=2.5):
        self.dataset = dataset
        self.beta = beta
        self.tau=tau
        self.kappa=kappa
        self.cls_num=cls_num
        self.lam = np.random.beta(self.beta, self.beta)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]

        # Randomly choose another sample
        idx2 = np.random.randint(0, len(self.dataset))
        x2, y2 = self.dataset[idx2]

        # Use the precomputed lambda
        lam = self.lam

        # Create new sample
        x = lam * x1 + (1 - lam) * x2
        y1 = torch.tensor(y1, dtype=torch.long)
        y2 = torch.tensor(y2, dtype=torch.long)
        #print(y1)
        # 找到样本最多的类
        #print("y1:", y1)
        # print("self.cls_num length:", self.cls_num)

        num_class1=self.cls_num[y1%10]
        num_class2 = self.cls_num[y2%10]
        # Determine lambda for labels
        if num_class1 / num_class2 >= self.kappa and lam < self.tau:
            lam_y = 0
        elif num_class2 / num_class1 >= self.kappa and (1 - lam) < self.tau:
            lam_y = 1
        else:
            lam_y = lam
        mixed_y = lam_y * y1 + (1 - lam_y) * y2
        lam_y=0
        # y = [y1, y2, lam_y]
        # y=torch.tensor(y)
        # return torch.tensor(x),  y
        return torch.tensor(x),  (y1, y2, lam_y)

    def set_lambda(self):
        self.lam = np.random.beta(self.beta, self.beta)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    labels_a, labels_b, lam = zip(*targets)
    labels_a = torch.stack(labels_a)
    labels_b = torch.stack(labels_b)
    lam = torch.tensor(lam)
    #print(lam)
    return inputs, (labels_a, labels_b, lam[0])

def create_dataloader(dataset, cls_num,batch_size, beta=1.0, shuffle=True, num_workers=0):
    remix_dataset = RemixDataset(dataset,cls_num ,beta)
    dataloader = DataLoader(remix_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader, remix_dataset


