
import os
import torch
from PIL import Image
import openvino as ov
from openvino.tools import mo
from openvino.runtime import serialize
import torchvision.transforms as transforms

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

"""
OpenVINO™ 是 Intel 推出的深度学习推理优化工具，支持多种硬件加速。通过 Model Optimizer 工具，可以将 ONNX 格式的模型转换为 OpenVINO IR（Intermediate Representation）格式，以便在 OpenVINO 运行时高效部署。

1. 安装 OpenVINO:
    pip install openvino-dev

2. 验证安装:
    mo --version  (如果提示找不到 `mo` 命令，可以用 `python -m openvino.tools.mo` 代替)

3. 使用 Model Optimizer 转换模型(假设已有 `model.onnx` 文件):
    mo  --input_model Best_LeNet_epoch_2.onnx \
        --input_shape "[1,3,224,224]" \
        --mean_values="[123.675,116.28,103.53]" \
        --scale_values="[58.395,57.12,57.375]" \
        --output_dir ir_output
    推荐使用 使用 Python API 进行转换。代码见本文件中可执行代码。

    常用参数说明:
    --input_model：输入的 ONNX 模型路径
    --input_shape：指定输入 shape（如 [1,3,224,224]）
    --mean_values / --scale_values: 指定归一化参数
    --output_dir：指定输出目录（默认当前目录）

4. 转换结果:
    model.xml：网络结构描述文件
    model.bin：权重文件

5. 验证转换结果:
    from openvino.runtime import Core
    ie = Core()
    model = ie.read_model(model="openvino_model/model.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    print("OpenVINO 模型加载成功！")

6. 量化模型:
    下载并解压花分类数据集，将 quantization_int8.py 中的 data_path 指向解压后的 flower_photos,
    使用 quantization_int8.py 量化模型.
"""

if __name__ == "__main__":

    root_dir = "deploying/convert_openvino/ir_output/"
    ir_model_xml = root_dir + "model.xml"
    ir_model_bin = root_dir + "model.bin"

    # 转换模型
    os.makedirs(os.path.dirname(root_dir), exist_ok=True)
    ov_model = ov.convert_model("deploying/convert_openvino/Best_LeNet_epoch_2.onnx",
                                input=[[1, 3, 224, 224]] # 可选指定形状
    )
    ov.save_model(ov_model, ir_model_xml)
    print("模型转换成功！")


    
    # ###############
    # 加载模型 & 推理
    from openvino.runtime import Core        
    core = Core() # 加载OpenVINO推理引擎
    net = core.read_model(model=ir_model_xml, weights=ir_model_bin) # 读取 IR 模型
    compiled_net = core.compile_model(model=net, device_name="CPU")  # 编译模型为可执行网络 "GPU"

    # 获取支持数据类型
    cpu_optimization_capabilities_device = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    print("CPU optimization capabilities:", cpu_optimization_capabilities_device)
    gpu_optimization_capabilities_device = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
    print("GPU optimization capabilities:", gpu_optimization_capabilities_device)
    
    # input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    m_input = None
    test_image_path = "D:/Users/74055/Desktop/OIP-C1.jpg"  # 可选
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

    # 执行推理(法一：默认)
    result = compiled_net(m_input, share_inputs=True, share_outputs=True)
    result = torch.from_numpy(result[0]).squeeze()
    # 执行推理2(法二：指定输入输出)
    inputs_names = compiled_net.inputs
    outputs_names = compiled_net.outputs
    request = compiled_net.create_infer_request()
    request.infer(inputs={inputs_names[0]: m_input})
    result = request.get_output_tensor(outputs_names[0].index).data


    # 查看结果，计算softmax概率
    classes = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')  # Flower5 dataset classes
    predict_cla = torch.argmax(result).item()
    print(f"Predicted class: {classes[predict_cla]}, Confidence scores: {result[predict_cla]}")