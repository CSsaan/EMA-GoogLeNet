import os
import torch
import argparse
import numpy as np
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import dataLoader.keyPointsTransform as transforms
from KeyPoint.model import create_DeepPose_model
from dataLoader.WFLW import draw_keypoints  # 加载WFLW数据集

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    img_path = args.input_path
    weights_path = args.model_path
    input_size = [args.input_size, args.input_size]
    num_keypoints = args.num_keypoints
    save_path = args.save_path
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.AffineTransform(scale_prob=0., rotate_prob=0., shift_prob=0., fixed_size=input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = np.array(Image.open(img_path))
    h, w, c = img.shape
    target = {"box": [0, 0, w, h]}
    img_tensor, target = transform(img, target=target)
    # expand batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # create model
    net = create_DeepPose_model(num_keypoints=num_keypoints).to(device)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, weights_only=True)["net"])

    # prediction
    net.eval()
    with torch.inference_mode():
        # with torch.autocast(device_type=device.type):
        pred = torch.squeeze(net(img_tensor.to(device))).reshape([-1, 2]).cpu().numpy()

        wh_tensor = np.array(input_size[::-1], dtype=np.float32).reshape([1, 2])
        pred = pred * wh_tensor  # rel coord to abs coord
        pred = transforms.affine_points_np(pred, target["m_inv"].numpy())
        draw_keypoints(img, coordinate=pred, save_path=save_path, radius=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/DeepPose/DeepPose_net_optimizer_scheduler_epoch31.pth', type=str, help='path to the model')
    parser.add_argument('--input_path', default= "/home/cs/test_img.jpg", type=str, help='image path for inference')
    parser.add_argument('--input_size', default= 256, type=int, help='input size for inference')
    parser.add_argument('--num_keypoints', default= 98, type=int, help='number of keypoints')
    parser.add_argument('--save_path', default= "./KeyPoint/predict.jpg", type=str, help='path to save the result')
    args = parser.parse_args()

    main(args)