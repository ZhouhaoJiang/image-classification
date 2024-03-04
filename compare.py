import onnxruntime as ort
import numpy as np
import torch
import torchvision.models as models
from torch import nn

# 加载和准备 PyTorch 模型
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 4)
model.load_state_dict(torch.load(r'./model/epoch_15_model.pth', map_location='cpu'))
model.eval()

# 准备输入数据
input_data = torch.randn(1, 3, 224, 224)

# 获取 PyTorch 模型输出
pytorch_output = model(input_data)

# 加载 ONNX 模型并获取输出
ort_session = ort.InferenceSession(r"./onnx_model/chat.onnx")
onnx_input = {ort_session.get_inputs()[0].name: input_data.numpy()}
onnx_output = ort_session.run(None, onnx_input)

# 打印输出以进行比较
print("PyTorch Model Output:", pytorch_output.detach().numpy())
print("ONNX Model Output:", onnx_output[0])

# 比较输出
try:
    np.testing.assert_allclose(pytorch_output.detach().numpy(), onnx_output[0], rtol=1e-03, atol=1e-05)
    print("The outputs are close enough.")
except AssertionError as e:
    print("The outputs differ:", e)