# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from model import GoogLeNet # 加载模型
from dataLoader.Flower5 import get_flower5_dataloaders  # 加载数据集
from config import load_model_parameters  # 加载模型参数配置

def evaluate(net, loss_function, val_image, val_label):
    net.eval()
    with torch.no_grad():
        val_outputs = net(val_image)
        val_loss = loss_function(val_outputs, val_label).item()
        predict_y = torch.max(val_outputs, dim=1)[1]
        accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
    return val_loss, accuracy

def main(parameters_file_path):
    """ Main function to train the GoogLeNet model on Flower5 dataset.
    Args: check in file: benchmark/config/GoogLeNet_parameters.yaml
    """
    # 0. Load parameters
    parameters = load_model_parameters(parameters_file_path)
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    input_size = parameters['input_size']
    num_workers = parameters['num_workers']
    learning_rate = parameters['learning_rate']
    num_classes = parameters['num_classes']
    print_every_minibatch = parameters['print_every_minibatch']
    dataset_path = parameters['dataset_path']
    save_path = parameters['save_path']
    os.makedirs(save_path, exist_ok=True)
    print(f"Using parameters from {parameters_file_path}:")
    print(f"Training GoogLeNet for {epochs} epochs, batch size:{batch_size}, learning rate:{learning_rate}, num_classes:{num_classes}, print every {print_every_minibatch} mini-batches, saving to:{save_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. DataLoader (Flower5 数据集加载)
    train_loader, val_loader, _trainset, _testset = get_flower5_dataloaders(data_dir=dataset_path, input_size=input_size, batch_size=batch_size, num_workers=num_workers)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter) #  classes: ('daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
    val_image, val_label = val_image.to(device, non_blocking=True), val_label.to(device, non_blocking=True)

    # 2. Initialize model (GoogLeNet)
    net = GoogLeNet(num_classes=num_classes)  # Flower5 has 3 channels (RGB) and 5 classes
    net = net.cuda() if torch.cuda.is_available() else net  # Check if GPU is available

    # 3. Define loss function
    loss_function = nn.CrossEntropyLoss()
    # 4. Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    

    # 5. Training loop
    min_val_loss = float('inf')  # 初始化最小验证损失

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # forward + backward + optimize
            optimizer.zero_grad() # zero the parameter gradients
            net.train()
            outputs = net(inputs) # forward
            loss = loss_function(outputs, labels)
            loss.backward() # backward
            optimizer.step() # optimize

            # print statistics
            running_loss += loss.item()
            if step % print_every_minibatch == 0:    # print every 500 mini-batches
                val_loss, accuracy = evaluate(net, loss_function, val_image, val_label)
                print('\r[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, step + 1, running_loss / print_every_minibatch, accuracy), end='', flush=True)
                running_loss = 0.0
                # 保存最小loss的模型
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(net.state_dict(), f"{save_path}/Best_GoogLeNet_epoch_{epoch + 1}.pth")
                    print(f"Model saved at epoch {epoch + 1}, step {step + 1} with val_loss: {val_loss:.3f}")
        # 每个epoch结束后，保存模型
        torch.save(net.state_dict(), f"{save_path}/GoogLeNet_epoch_{epoch + 1}.pth")
    print('Finished Training')

    torch.save(net.state_dict(), f"{save_path}/GoogLeNet_final.pth")



if __name__ == '__main__':
    # 加载模型参数配置
    ALL_parameters_file_path = 'benchmark/config/GoogLeNet_parameters.yaml'

    main(ALL_parameters_file_path)