# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_flower5_dataloaders(data_dir='./dataset/Flower5', input_size=224, batch_size=32, num_workers=2):
    """
    加载花蕊分类数据集的dataloader
    """
    data_transform = transforms.Compose([
        # transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据集
    test_ratio = 0.2  # 测试集比例
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transform)

    # 划分训练集和测试集
    dataset_size = len(full_dataset)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 创建数据加载器
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # # 类别映射
    class_to_idx = full_dataset.class_to_idx  # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    print(class_to_idx)

    return trainloader, testloader, train_dataset, test_dataset


# 用法示例
if __name__ == "__main__":
    data_dir = "./dataset/Flower5"  # 替换为你的数据集路径
    trainloader, testloader, train_dataset, test_dataset = get_flower5_dataloaders(data_dir)
    
    # for imgs, labels in trainloader:
    #     print(imgs.shape, labels)
    #     break