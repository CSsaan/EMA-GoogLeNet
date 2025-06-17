from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from model import GoogLeNet


"""
ONNX（Open Neural Network Exchange）是一种开放的深度学习模型交换格式，支持多种框架之间的模型互操作。
PyTorch 支持将模型导出为 ONNX 格式，便于在其他平台（如 TensorRT、ONNX Runtime 等）部署。
"""

device = torch.device("cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def export_model(model_weight_path, save_path):
    """导出PyTorch模型为ONNX格式"""
    assert isinstance(save_path, str), "lack of save_path parameter..."
    
    # 创建并加载模型
    model = GoogLeNet(num_classes=5)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # 准备输入张量
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    
    # 导出模型
    torch.onnx.export(model,                     # 要导出的模型
                      x,                         # 模型输入
                      save_path,                 # 保存路径
                      export_params=True,        # 是否导出模型参数
                      opset_version=11,          # ONNX算子集版本（建议使用 11 或更高）
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],     # 输入名称
                      output_names=["output"],   # 输出名称
                      dynamic_axes={             # 动态轴（可选）
                          "input": {0: "batch_size"}, 
                          "output": {0: "batch_size"}
                      })
    print(f"[Model exported successfully to: {save_path}]")


def test_model(model_path, test_image_path=None):
    """测试导出的ONNX模型"""
    # 检查ONNX模型
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX model checked successfully!]")
    
    # 创建ONNX运行时会话
    ort_session = onnxruntime.InferenceSession(model_path)
    
    m_input = None
    if test_image_path:
        # 加载并预处理图像
        m_input = Image.open(test_image_path)
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        m_input = transform(m_input).unsqueeze_(0)
    else:
        # 测试随机输入
        m_input = torch.rand(1, 3, 224, 224)

    # 运行模型
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(m_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = ort_outs[0]
    print("[ONNX model test with input passed!]")
        
    # 查看结果，计算softmax概率
    classes = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')  # Flower5 dataset classes
    prediction -= np.max(prediction, keepdims=True)
    prediction = np.exp(prediction) / np.sum(np.exp(prediction), keepdims=True)
    predict_cla = np.argmax(prediction)
    print(f" Predicted class: [{classes[predict_cla]}], Prediction probabilities: {prediction.squeeze()}")

    # prediction = torch.from_numpy(prediction)
    # predict_cla = torch.argmax(prediction).item()
    # print(f"Predicted class: {classes[predict_cla]}, Confidence scores: {prediction[predict_cla]}")
    return ort_session


if __name__ == '__main__':
    # 配置参数
    model_weight_path = "checkpoints/GoogLeNet/Best_LeNet_epoch_2.pth"
    onnx_file_name = "deploying/convert_onnx/" + model_weight_path.split("/")[-1].split(".")[0] + ".onnx"
    test_image_path = "D:/Users/74055/Desktop/OIP-C1.jpg"  # 可选
    
    # 导出模型
    export_model(model_weight_path, onnx_file_name)
    
    # 测试模型
    test_model(onnx_file_name, test_image_path)