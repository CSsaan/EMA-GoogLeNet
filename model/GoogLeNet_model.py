import torch
from torch import nn
from torchsummary import summary

# netron模型可视化
import netron
import torch.onnx
from torch.autograd import Variable

# import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()

        # 路线1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)

        # 路线2，1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)

        # 路线3，1×1卷积层, 5×5的卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)

        # 路线4，3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)



class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes))

        if init_weights:
            self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x
        


if __name__ == "__main__":
    # 打印模型网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet().to(device)
    print(summary(model, (3, 224, 224)))

    # 可视化模型网络结构
    input_x = torch.randn(8, 3, 224, 224).to(device)  # 随机生成一个输入
    modelData = "./demo.pth"  # 定义模型数据保存的路径
    # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
    torch.onnx.export(model, input_x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    netron.start(modelData)  # 输出网络结构



































# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=1000, aux_logits=False, init_weights=True):
#         super(GoogLeNet, self).__init__()
#         self.aux_logits = aux_logits

#         self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.conv2 = BasicConv2d(64, 64, kernel_size=1)
#         self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

#         self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

#         if self.aux_logits:
#             self.aux1 = InceptionAux(512, num_classes)
#             self.aux2 = InceptionAux(528, num_classes)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)
#         if init_weights:
#             self._initialize_weights()

#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.conv1(x)
#         # N x 64 x 112 x 112
#         x = self.maxpool1(x)
#         # N x 64 x 56 x 56
#         x = self.conv2(x)
#         # N x 64 x 56 x 56
#         x = self.conv3(x)
#         # N x 192 x 56 x 56
#         x = self.maxpool2(x)

#         # N x 192 x 28 x 28
#         x = self.inception3a(x)
#         # N x 256 x 28 x 28
#         x = self.inception3b(x)
#         # N x 480 x 28 x 28
#         x = self.maxpool3(x)
#         # N x 480 x 14 x 14
#         x = self.inception4a(x)
#         # N x 512 x 14 x 14
#         if self.training and self.aux_logits:    # eval model lose this layer
#             aux1 = self.aux1(x)

#         x = self.inception4b(x)
#         # N x 512 x 14 x 14
#         x = self.inception4c(x)
#         # N x 512 x 14 x 14
#         x = self.inception4d(x)
#         # N x 528 x 14 x 14
#         if self.training and self.aux_logits:    # eval model lose this layer
#             aux2 = self.aux2(x)

#         x = self.inception4e(x)
#         # N x 832 x 14 x 14
#         x = self.maxpool4(x)
#         # N x 832 x 7 x 7
#         x = self.inception5a(x)
#         # N x 832 x 7 x 7
#         x = self.inception5b(x)
#         # N x 1024 x 7 x 7

#         x = self.avgpool(x)
#         # N x 1024 x 1 x 1
#         x = torch.flatten(x, 1)
#         # N x 1024
#         x = self.dropout(x)
#         x = self.fc(x)
#         # N x 1000 (num_classes)
#         if self.training and self.aux_logits:   # eval model lose this layer
#             return x, aux2, aux1
#         return x

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


# class Inception(nn.Module):
#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super(Inception, self).__init__()

#         self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channels, ch3x3red, kernel_size=1),
#             BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
#         )

#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channels, ch5x5red, kernel_size=1),
#             # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
#             # Please see https://github.com/pytorch/vision/issues/906 for details.
#             BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
#         )

#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             BasicConv2d(in_channels, pool_proj, kernel_size=1)
#         )

#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)

#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)


# class InceptionAux(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
#         self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

#         self.fc1 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
#         x = self.averagePool(x)
#         # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
#         x = self.conv(x)
#         # N x 128 x 4 x 4
#         x = torch.flatten(x, 1)
#         x = F.dropout(x, 0.5, training=self.training)
#         # N x 2048
#         x = F.relu(self.fc1(x), inplace=True)
#         x = F.dropout(x, 0.5, training=self.training)
#         # N x 1024
#         x = self.fc2(x)
#         # N x num_classes
#         return x


# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x