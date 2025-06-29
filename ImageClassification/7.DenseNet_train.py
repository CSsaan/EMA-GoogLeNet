# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ImageClassification.model import densenet121, densenet169, densenet201, densenet161 # 加载模型
from dataLoader.Flower5 import get_flower5_dataloaders  # 加载数据集
from benchmark.utils.config import load_model_parameters  # 加载模型参数配置

"""
22. 花卉数据集: Flower5, 手动下载解压至./dataset/Flower5目录下。下载地址见doc/dataset_info.md。
"""

def main(parameters_file_path):
    """ Main function to train the DenseNet model on Flower5 dataset.
    Args: check in file: benchmark/config/DenseNet_parameters.yaml
    """
    # 0. Load parameters
    parameters = load_model_parameters(parameters_file_path)
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    input_size = parameters['input_size']
    num_workers = parameters['num_workers']
    learning_rate = parameters['learning_rate']
    num_classes = parameters['num_classes']
    dataset_path = parameters['dataset_path']
    save_path = parameters['save_path']
    os.makedirs(save_path, exist_ok=True)
    print(f"Using parameters from {parameters_file_path}:")
    print(f"Training DenseNet for {epochs} epochs, batch size:{batch_size}, learning rate:{learning_rate}, num_classes:{num_classes}, saving to:{save_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. DataLoader (Flower5 数据集加载)
    train_loader, val_loader, _trainset, _testset = get_flower5_dataloaders(data_dir=dataset_path, input_size=input_size, batch_size=batch_size, num_workers=num_workers)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter) #  classes: ('daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
    val_image, val_label = val_image.to(device, non_blocking=True), val_label.to(device, non_blocking=True)

    # 2. Initialize model (densenet121, densenet169, densenet201, densenet161)
    net = densenet121(num_classes=num_classes).to(device) # Flower5 has 3 channels (RGB) and 5 classes

    # 加载预训练权重：download url: 详见./model/DenseNet.py
    load_pretrained = False
    if load_pretrained:
        model_weight_path = "./checkpoints/DenseNet/densenet_xxx.pth"
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        pre_weights = torch.load(model_weight_path, weights_only=True)
        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        
    # 冻结部分权重
    freeze_layers = False
    if freeze_layers:
        for name, para in net.named_parameters():
            if "classifier" not in name: # 除最后的全连接层外，其他权重全部冻结
                para.requires_grad_(False)
        # 只训练分类器部分的参数比例
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"TrainableParams/TotalParams: {trainable_params:,}/{total_params:,}, {100 * trainable_params / total_params:.1f}%")

    # 3. Define loss function
    loss_function = nn.CrossEntropyLoss()
    # 4. Define optimizer
    params = [p for p in net.parameters() if p.requires_grad] # 除去冻结部分
    optimizer = optim.Adam(params, lr=learning_rate)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lrf = learning_rate * 10
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 5. Training loop
    best_acc = 0.0  # 初始化最佳准确率
    train_steps = len(train_loader)  # 计算每个epoch的迭代次数
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc='Training Progress')
        for step, data in enumerate(train_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
            # forward + backward + optimize
            optimizer.zero_grad() # zero the parameter gradientsResNe
            outputs = net(inputs) # forward
            loss = loss_function(outputs, labels)
            loss.backward() # backward
            optimizer.step() # optimize

            # sum up loss
            running_loss += loss.item()

            # 进度条显示
            postfix = {
                'progress': '[{}/{}]'.format(epoch + 1, epochs),
                'lr': '{:.6f}'.format(optimizer.param_groups[0]['lr']),
                'loss': '{:.4f}'.format(loss)
            }
            train_bar.set_postfix(postfix)
        scheduler.step() # 更新学习率
        print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))

        
        
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        loss = 0.0
        val_num = len(_testset)
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validating Progress')
            for val_data in val_bar:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device, non_blocking=True), val_labels.to(device, non_blocking=True) 

                val_outputs = net(val_inputs)
                loss += loss_function(val_outputs, val_labels).item()
                predict_y = torch.max(val_outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        val_loss = loss / val_num
        print('val_loss: %.3f  val_accuracy: %.3f' % (val_loss, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), f"{save_path}/Best_DenseNet_epoch_{epoch + 1}.pth")
            print(f"Model saved at best accuracy: {best_acc:.3f}")

        # 每个epoch结束后，保存模型
        torch.save(net.state_dict(), f"{save_path}/DenseNet_epoch_{epoch + 1}.pth")
    
    print('Finished Training')
            
        

if __name__ == '__main__':
    # 加载模型参数配置
    ALL_parameters_file_path = 'benchmark/config/DenseNet_parameters.yaml'

    main(ALL_parameters_file_path)