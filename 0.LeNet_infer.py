import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet # 加载模型


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main(args):
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # Resize to 32x32
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = LeNet(num_classes=10).to(device)
    net.load_state_dict(torch.load(args.model_path))

    im = Image.open(args.input_path).convert('RGB')  # Convert image to RGB
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        im = im.to(device, non_blocking=True)
        net.eval()
        outputs = net(im)
        predict = torch.max(outputs if device == 'cpu' else outputs.cpu(), dim=1)[1].numpy()
    print(classes[int(predict.item())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/LeNet/Best_LeNet_epoch_4.pth', type=str, help='path to the model')
    parser.add_argument('--input_path', default= "D:/Users/74055/Desktop/OIP-C.jpg", type=str, help='image path for inference')
    args = parser.parse_args()

    main(args)