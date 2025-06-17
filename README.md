# Image Classification
Using EMA to train Image Classification task.

| 神经网络 | 年份 | 标签 | 作者 |
| --- | --- | --- | --- |
| [LeNet](https://ieeexplore.ieee.org/document/726791) | 1998年 | CNN开山之作 | 纽约大学 |
| [AlexNet](https://arxiv.org/abs/AlexNet) | 2012年 | 深度学习 CV领域划时代论文 具有里程碑意义 ImageNet 2020冠军 | 多伦多大学 Hinton团队 |
| [GoogLeNet](https://arxiv.org/pdf/1409.4842) | 2014年 | Google系列论文开创论文 （ImageNet 大规模视觉识别挑战赛 2014冠军 Inception模块 | 谷歌 |
| **VGG** | 2014年 | 开启3*3卷积堆叠时代 ImageNet 2014亚军 VGG-16和VGG-19 | 牛津大学 |
| [ResNet](https://arxiv.org/abs/1512.03385) | 2015年 | 最具影响力的卷积 神经网络 ImageNet 2015冠军 残差网络 | 何凯明团队 微软亚院 |
| **DenseNet** | 2017年 | ImageNet 2016冠军 CVPR 2017最佳论文 Dense模块 | 康奈尔大学 清华大学 |
| [MobileNet](https://arxiv.org/abs/1704.04861) | 2017年 | 轻量级 Group卷积 Depthwise Seperable卷积 | 谷歌 |
| [ShuffleNetV2](https://arxiv.org/abs/1807.11164) | 2018年 | ImageNet 2018冠军 旷视科技 | 旷视科技 |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | 2019年 | ImageNet 2019冠军 使用复合缩放方法来平衡网络的宽度、深度和分辨率 | 谷歌 |

## 📦 Requirements
Python >= 3.8
torch >= 1.8.0
CUDA Version >= 11.7
skimage 0.19.2
numpy 1.23.1
opencv-python 4.6.0
timm 0.6.11
pillow
tqdm
onnx
openvino-dev
nncf
torchsummary


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
├── benchmark
│   ├── utils
│   └── config                    -> all model's parameters
├── dataLoader                    -> dataLoader for each dataset.
├── dataset                       -> download dataset and put it in this folder.
├── deploying
│   ├── convert_onnx              -> convert pytorch model to onnx.
│   ├── convert_tensorRT          -> convert pytorch model to tensorrt.
│   └── convert_openvino          -> convert pytorch model to openvino.
├── model_train.py                -> load model & train.
├── model_infer.py                -> load model & inference.
├── config.py                     -> some configurations.
├── requirements.txt
├── log                           -> tensorboard log.
└── model                         -> model definition.
    ├── lenet.py
    ├── alexnet.py
    ├── googlenet.py
    ├── resnet.py
    ├── mobilenet.py
    ├── shufflenet.py
    └── ... .py
```
