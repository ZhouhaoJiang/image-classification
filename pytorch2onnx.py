#coding: utf-8
import torch
from torch import nn
#import torchvision
# 1.导入pytorch模型定义
from torchvision import models

# 2.指定输入大小的shape
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 构建pytorch model
model = models.mobilenet_v2(pretrained=True)  # 使用预训练的 MobileNetV2
model.classifier[1] = nn.Linear(model.last_channel, 4)  # 替换分类层以适应二分类

# 4. 载入模型参数
model.load_state_dict(torch.load('./model/epoch_10_model.pth', map_location='cpu'))
model.eval()

# 5.导出onnx模型文件
torch.onnx.export(model, dummy_input, "./onnx_model/chat.onnx",verbose=True)