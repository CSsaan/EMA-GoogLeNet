# Image Classification
Using EMA to train Image Classification task.

 - [LeNet](https://ieeexplore.ieee.org/document/726791):最早的卷积神经网络之一。
 - [AlexNet](https://arxiv.org/abs/AlexNet):在 ILSVRC 2012 中取得了第一名。
 - [GoogLeNet](https://arxiv.org/pdf/1409.4842):在 ILSVRC 2014（ImageNet 大规模视觉识别挑战赛）中取得了第一名。
 - [ResNet](https://arxiv.org/abs/1512.03385):在 ILSVRC 2015 中取得了第一名。
 - [MobileNet](https://arxiv.org/abs/1704.04861):2017年在移动设备上运行的轻量级卷积神经网络。
 - [ShuffleNet](https://arxiv.org/abs/1707.01083):2017年在移动设备上运行的轻量级卷积神经网络，使用通道混洗操作。
 - [EfficientNet](https://arxiv.org/abs/1905.11946):在 ILSVRC 2019 中取得了第一名，使用复合缩放方法来平衡网络的宽度、深度和分辨率。

## 📦 Requirements
Python >= 3.8
torch >= 1.8.0
CUDA Version >= 11.7
skimage 0.19.2
numpy 1.23.1
opencv-python 4.6.0
timm 0.6.11
tqdm

## 📦 Installation
```bash
git clone git@github.com:CSsaan/EMA-GoogLeNet.git
cd EMA-GoogLeNet
conda create -n EMA python=3.10 -y
conda activate EMA
pip install -r requirements.txt
```

## Dataset

- 本工程以[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)为例进行图像分类任务。
- 其他数据集可参考[Image Classification Datasets](doc\dataset_info.md)

## 📖 Usage

1.0. Download dataset and put it in `data` folder.
2.0. Edit `config.py` to set the model and dataset you want to use.

## 📂 Repo structure (WIP)
```
├── README.md
├── Trainer.py                    -> load model & train.
├── config.py                     -> all models dictionary
├── dataset.py                    -> dataLoader
├── demo_Fc.py                    -> model inder
├── pyproject.toml                ->  project config
├── requirements.txt
├── train.py                      -> main
├── benchmark
│   ├── loss.py
│   └── config                    -> all model's parameters
└── utils
│   ├── testGPU.py
│   ├── yuv_frame_io.py
│   └── print_structure.py
├── data                          -> dataset
├── log                           -> tensorboard log
└── model
```
