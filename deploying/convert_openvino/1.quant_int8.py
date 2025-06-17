import nncf
import torch
from PIL import Image
from openvino.runtime import serialize
from openvino.tools import mo
from openvino.runtime import Core
import torchvision.transforms as transforms

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from dataLoader.Flower5 import get_flower5_dataloaders  # 加载数据集


def quantize_model(ir_model_xml, ir_model_bin, nncf_int8_xml, nncf_int8_bin, data_dir='./dataset/Flower5', input_size=224, batch_size=1, num_workers=4):
    """
    量化模型并保存为 INT8 格式。
    """
    # 加载原始模型
    core = Core()
    ov_model = core.read_model(model=ir_model_xml, weights=ir_model_bin)
    # 数据预处理
    def transform_fn(data_item):
        return data_item[0]
    # 创建数据集
    train_loader, val_loader, _, _ = get_flower5_dataloaders(data_dir=data_dir, input_size=input_size, batch_size=batch_size, num_workers=num_workers)
    nncf_calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    # 量化模型
    quantized_model = nncf.quantize(ov_model, nncf_calibration_dataset, preset=nncf.QuantizationPreset.MIXED, subset_size=300)
    # 保存量化后的模型
    serialize(quantized_model, nncf_int8_xml, nncf_int8_bin)
    print("Quantization completed and saved.")

def run_inference(nncf_int8_xml, nncf_int8_bin, test_image_path=None):
    """
    加载量化后的模型并进行推理。
    """
    # 加载模型
    core = Core()
    net = core.read_model(model=nncf_int8_xml, weights=nncf_int8_bin) # 读取 IR 模型
    compiled_net = core.compile_model(model=net, device_name="CPU")   # 编译模型为可执行网络 "GPU"
    # 获取支持数据类型
    cpu_optimization_capabilities_device = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    print("CPU optimization capabilities:", cpu_optimization_capabilities_device)
    gpu_optimization_capabilities_device = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
    print("GPU optimization capabilities:", gpu_optimization_capabilities_device)
    # 准备输入数据
    m_input = None
    if test_image_path:
        # 加载并预处理图像
        m_input = Image.open(test_image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        m_input = transform(m_input).unsqueeze_(0).numpy()
    else:
        # 测试随机输入
        m_input = torch.rand(1, 3, 224, 224)

    # 执行推理(法一：默认)
    result = compiled_net(m_input, share_inputs=True, share_outputs=True)
    result = torch.from_numpy(result[0]).squeeze()
    # 执行推理2(法二：指定输入输出)
    inputs_names = compiled_net.inputs
    outputs_names = compiled_net.outputs
    request = compiled_net.create_infer_request()
    request.infer(inputs={inputs_names[0]: m_input})
    result = request.get_output_tensor(outputs_names[0].index).data

    # 查看结果，计算 softmax 概率
    classes = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')  # Flower5 dataset classes
    predict_cla = torch.argmax(result).item()
    print(f"Predicted class: {classes[predict_cla]}, Confidence scores: {result[predict_cla]}")



if __name__ == "__main__":
    
    root_dir = "deploying/convert_openvino/ir_output/"
    ir_model_xml = root_dir + "model.xml"
    ir_model_bin = root_dir + "model.bin"
    nncf_int8_xml = root_dir + "model_int8.xml"
    nncf_int8_bin = root_dir + "model_int8.bin"

    # 量化模型
    quantize_model(ir_model_xml, ir_model_bin, nncf_int8_xml, nncf_int8_bin)

    # 推理
    test_image_path = "D:/Users/74055/Desktop/OIP-C1.jpg"
    run_inference(nncf_int8_xml, nncf_int8_bin, test_image_path=test_image_path)