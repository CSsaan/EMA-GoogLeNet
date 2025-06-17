import os
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import YourModel
from dataset import YourDataset


def setup():                                                                            # [XXX]: parallel_torchrun
    dist.init_process_group(backend='nccl')  # CPU用'gloo', GPU用'nccl'
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # 获取排名和设备
    local_rank = int(os.environ['LOCAL_RANK'])  # 本地排名
    rank = int(os.environ['RANK'])  # 全局排名
    return local_rank, rank


def main():

    # 设置分布式环境
    local_rank, rank = setup()                                                          # [XXX]: setup() 

    device = torch.device(f'cuda:{local_rank}')

    # 定义模型并移动到指定 GPU
    model = YourModel().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])         # [XXX]: n.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 数据加载器
    train_dataset = YourDataset()
    train_sampler = DistributedSampler(train_dataset)                                   # [XXX]: DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    # 训练循环
    for epoch in range(10):
        train_sampler.set_epoch(epoch)                                                  # [XXX]: train_sampler.set_epoch(epoch)  # 确保每个 epoch 数据洗牌不同
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # 训练步骤...
            if rank == 0:                                                               # [XXX]: if rank == 0                    # 仅主进程保存模型或打印日志
                print(f"Epoch {epoch}, Loss: ...")

    # 清理进程组
    dist.destroy_process_group()                                                        # [XXX]: dist.destroy_process_group() 

if __name__ == '__main__':
    main()
