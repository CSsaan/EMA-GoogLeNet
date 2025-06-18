import os
import math
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import dataLoader.keyPointsTransform as transforms
from KeyPoint.model import create_DeepPose_model
from dataLoader.WFLW import get_WFLW_dataloaders  # 加载WFLW数据集
from benchmark.utils.config import load_model_parameters  # 加载模型参数配置
from benchmark.DeepPose_losses import WingLoss, NMEMetric
    

def main(parameters_file_path):
    torch.manual_seed(1234)
    
    # 0. Load parameters
    parameters = load_model_parameters(parameters_file_path)
    epochs = parameters['epochs']
    lr = float(parameters['lr'])
    input_size = [parameters['input_size'], parameters['input_size']]
    num_keypoints = parameters['num_keypoints']
    epochs = parameters['epochs']
    start_epoch = 0
    batch_size = parameters['batch_size']
    num_workers = parameters['num_workers']
    warmup_epoch = parameters['warmup_epoch']
    lr_steps = parameters['lr_steps']
    dataset_path = parameters['dataset_path']
    save_path = parameters['save_path']
    resume = parameters['resume']
    os.makedirs(save_path, exist_ok=True)
    print(f"Using parameters from {parameters_file_path}:")
    print(f"Training EfficientNet for {epochs} epochs, batch size:{batch_size}, learning rate:{lr}, saving to:{save_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. DataLoader (Flower5 数据集加载)
    train_loader, val_loader, _trainset, _testset = get_WFLW_dataloaders(data_dir=dataset_path, input_size=input_size[0], batch_size=batch_size, num_workers=num_workers)

    # 2. Initialize model
    net = create_DeepPose_model(num_keypoints).to(device)

    # 3. Define loss function
    loss_function = WingLoss().to(device)

    # 4. Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Define learning rate Scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=len(train_loader) * warmup_epoch)
    multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[len(train_loader) * i for i in lr_steps],
            gamma=0.1)

    scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, multi_step_scheduler])

    if resume:
        assert os.path.exists(resume)
        checkpoint = torch.load(resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(start_epoch))

    # 5. Training loop
    best_acc = 0.0  # 初始化最佳准确率
    train_steps = len(train_loader)  # 计算每个epoch的迭代次数
    for epoch in range(start_epoch, epochs):

        # train
        net.train()
        wh_tensor = torch.as_tensor(input_size[::-1], dtype=torch.float32, device=device).reshape([1, 1, 2])
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc='Training Progress')
        for step, (inputs, targets) in enumerate(train_bar):
            inputs = inputs.to(device)
            labels = targets["keypoints"].to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type): # use mixed precision to speed up training
                pred: torch.Tensor = net(inputs)
                loss: torch.Tensor = loss_function(pred.reshape((-1, num_keypoints, 2)), labels, wh_tensor)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            loss.backward()
            optimizer.step()

            # sum up loss
            running_loss += loss.item()

            # 进度条显示
            postfix = {
                'progress': '[{}/{}]'.format(epoch + 1, epochs),
                'lr': '{:.6f}'.format(optimizer.param_groups[0]['lr']),
                'loss': '{:.4f}'.format(loss)
            }
            train_bar.set_postfix(postfix)   
        scheduler.step()
        print('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / train_steps))


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        loss = 0.0
        metric = NMEMetric(device=device)
        wh_tensor = torch.as_tensor(input_size[::-1], dtype=torch.float32, device=device).reshape([1, 1, 2])
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, desc='Validating Progress')
            for step, (inputs, targets) in enumerate(val_bar):
                inputs = inputs.to(device)
                m_invs = targets["m_invs"].to(device)
                labels = targets["ori_keypoints"].to(device)

                pred = net(inputs)
                pred = pred.reshape((-1, num_keypoints, 2))  # [N, K, 2]
                pred = pred * wh_tensor  # rel coord to abs coord
                pred = transforms.affine_points_torch(pred, m_invs)
                metric.update(pred, labels) # sum up NME

        nme = metric.evaluate()
        print(f"evaluation NME[{epoch}]: {nme:.3f}")

        # save weights
        save_files = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        if nme > best_acc:
            best_acc = nme
            torch.save(save_files, os.path.join(save_path, f"Best_DeepPose_net_optimizer_scheduler_epoch{epoch}.pth"))
            torch.save(net.state_dict(), f"{save_path}/Best_DeepPose_epoch_{epoch + 1}.pth")
            print(f"Model saved at best accuracy: {best_acc:.3f}")

        # 每个epoch结束后，保存模型
        torch.save(save_files, os.path.join(save_path, f"DeepPose_net_optimizer_scheduler_epoch{epoch}.pth"))
        torch.save(net.state_dict(), f"{save_path}/DeepPose_epoch_{epoch + 1}.pth")

if __name__ == '__main__':

    # 加载模型参数配置
    ALL_parameters_file_path = 'benchmark/config/DeepPose_parameters.yaml'

    main(ALL_parameters_file_path)