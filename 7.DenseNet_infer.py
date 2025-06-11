import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from model import densenet121, densenet169, densenet201, densenet161 # 加载模型


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')  # Flower5 dataset classes

def main(args):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),  # Resize to 224x224
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = densenet121(num_classes=5).to(device)
    net.load_state_dict(torch.load(args.model_path, weights_only=True))

    im = Image.open(args.input_path).convert('RGB')  # Convert image to RGB
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        im = im.to(device, non_blocking=True)
        net.eval()
        outputs = net(im)
        output = torch.squeeze(outputs) if device == 'cpu' else torch.squeeze(outputs).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).item()
        print(f"Predicted class: {classes[predict_cla]}")
        print(f"Confidence scores: {predict[predict_cla].item()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/DenseNet/Best_DenseNet_epoch_32.pth', type=str, help='path to the model')
    parser.add_argument('--input_path', default= "/home/cs/R-C.jpeg", type=str, help='image path for inference')
    args = parser.parse_args()

    main(args)