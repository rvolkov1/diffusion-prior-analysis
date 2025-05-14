import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class LongTailCIFAR100(Dataset):
    def __init__(self, root, transform=None, imb_factor=0.01, download=True):
        self.dataset = CIFAR100(root=root, train=True, download=download)
        self.transform = transform
        self.cls_num = 100
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k,v in self.class_to_idx.items()}

        # Generate long-tail distribution
        img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_factor)
        self.data, self.targets = self.gen_imbalanced_data(img_num_per_cls)

    def get_img_num_per_cls(self, cls_num, imb_factor):
        """Generate the number of samples per class following exponential decay"""
        max_num = len(self.dataset) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = max_num * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        """Subsample original dataset to create imbalance"""
        cls_data = defaultdict(list)
        for img, label in zip(self.dataset.data, self.dataset.targets):
            cls_data[label].append(img)

        new_data, new_targets = [], []
        for cls_idx, num in enumerate(img_num_per_cls):
            cls_imgs = cls_data[cls_idx]
            random.shuffle(cls_imgs)
            selected_imgs = cls_imgs[:num]
            new_data.extend(selected_imgs)
            new_targets.extend([cls_idx] * num)

        return new_data, new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        if self.transform:
            from PIL import Image
            img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = LongTailCIFAR100(root="./data",transform=transform,imb_factor=0.1)
    class_idx = 0
    counts = [0]*100
    print(len(counts))
    for sample in iter(dataset):
        sample_label_idx = sample[1]
        counts[sample_label_idx]+=1
        if class_idx==sample_label_idx:
            sampling.show_image(sample[0],dataset.idx_to_class[sample[1]])
            class_idx+=1

    print(counts)
    plt.figure()
    plt.hist(counts)
    plt.show()
