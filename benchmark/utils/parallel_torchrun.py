import os
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model import YourModel
from dataset import YourDataset


"""
torchrun 在 PyTorch 1.10 及以上版本可用。

torchrun 使用示例:
-----------------
- (1).单节点多 GPU 训练
    - 假设你有一个训练脚本 train.py，需要在单节点上的 4 个 GPU 上运行：
        torchrun --nproc_per_node=4/
                --nnodes=1 
                --node_rank=0 
                --rdzv_endpoint=localhost:1234 
                train.py 
    - 常用参数:      
        --nproc_per_node：每个节点上启动的进程数，通常设置为 GPU 数量（GPU 训练）或 CPU 核心数。
        --nnodes：参与训练的节点（机器）总数，范围可以是固定值或 min:max 格式（弹性模式）。
        --node_rank：当前节点的排名，范围是 0 到 nnodes-1。单节点时设为 0，多节点时每个节点需不同排名。
        --rdzv_endpoint：指定主节点地址和端口，确保未被占用。

- (2).多节点多 GPU 训练
    - 假设有 2 个节点，每个节点有 4 个 GPU，节点1 的IP为 192.168.1.1，节点2 的IP为 192.168.1.2：
        在节点 1（主节点）上运行：
            torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --rdzv_endpoint=192.168.1.1:1234 train.py
        在节点 2 上运行：
            torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --rdzv_endpoint=192.168.1.1:1234 train.py
        其中，主节点（node_rank=0）协调通信，其他节点连接到 rdzv_endpoint。

- (3).弹性训练
    - 支持动态节点数，例如允许 2 到 4 个节点参与：
        torchrun --nproc_per_node=4 --nnodes=2:4 --node_rank=0 --rdzv_endpoint=localhost:1234 --max_restarts=3 train.py
    - 参数差异:  
        --nnodes=2:4：节点数可在 2 到 4 之间动态调整。
        --max_restarts=3：允许失败后重启 3 次。

- (4).与 torch.distributed.launch 的区别
    - 命令简化：无需显式传递 --local_rank，torchrun 通过环境变量管理。
    - 弹性支持：torchrun 支持动态节点数和容错，launch 不支持。
    - 现代性：torch.distributed.launch 已弃用，torchrun 是官方推荐工具。
"""


def setup():                                                                            # [XXX]: parallel_torchrun
    dist.init_process_group(backend='nccl')  # CPU用'gloo', GPU用'nccl'
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # 获取排名
    local_rank = int(os.environ['LOCAL_RANK'])  # 本地排名：（0 到 nproc_per_node-1）
    rank = int(os.environ['RANK'])              # 全局排名：（0到world_size-1）

    world_size = int(os.environ['WORLD_SIZE'])  # 总进程数：（nnodes * nproc_per_node）
    master_addr = os.environ['MASTER_ADDR']     # 主节点地址
    master_port = os.environ['MASTER_PORT']     # 主节点端口

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
            # 注意！！：保存模型、日志等操作应仅在主进程（RANK == 0）执行，避免冲突。
            if rank == 0:                                                               # [XXX]: if rank == 0                    # 仅主进程保存模型或打印日志
                print(f"Epoch {epoch}, Loss: ...")

    # 清理进程组
    dist.destroy_process_group()                                                        # [XXX]: dist.destroy_process_group() 

if __name__ == '__main__':
    main()
