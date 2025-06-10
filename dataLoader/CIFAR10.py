# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms


# 定义数据预处理
def get_cifar10_loaders(root='./dataset/CIFAR10', input_size=32, batch_size=64, num_workers=2):
    """
    返回CIFAR-10的训练和测试数据加载器
    """
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 加载训练集 & 测试集
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=data_transform["train"])
    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=data_transform["val"])
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    print('Classes:', trainset.classes)
    return trainloader, testloader, trainset, testset



# 检查数据加载是否正常
def check_data():
    trainloader, testloader, trainset, testset = get_cifar10_loaders()
    # 检查训练集
    print(f'Training set size: {len(trainset)}')
    print(f'First training sample: {trainset[0]}')
    print('Classes:', trainset.classes) #  ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 检查测试集
    print(f'Test set size: {len(testset)}')
    print(f'First test sample: {testset[0]}')
    print('Classes:', testset.classes)

if __name__ == "__main__":
    check_data()
