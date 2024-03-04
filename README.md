# pytorch-train
该项目基于Pytorch，通过MobileNetV2网络结构，实现了聊天截图，身份票据，隐私图片的四分类任务，实现了模型的训练，测试，模型的保存，模型的加载转换，训练过程可视化等功能。
## 1. 项目结构
```
├── dataset
│   ├── train
│   │   ├── chat_records
│   │   ├── identity
│   │   ├── others
│   │   └── privacy
│   └── val
│       ├── chat_records
│       ├── identity
│       ├── others
│       └── privacy
├── onnx_model
├── pd_model  
```
其中identity数据集使用的是[HuggingFace的公开数据集](https://huggingface.co/datasets/erikaxenia/id_card_class_da)

## 2. 项目运行
### 环境要求
```
pip install -r requirements.txt
```
### 训练模型
准备好数据集后，运行
```
python train.py
```

## 3. 项目注意事项
x2paddle目前不支持arm架构的mac  
所以优先在Linux环境下进行模型转换  
windows环境下有概率出现模型转换失败或损坏的情况
具体参考:https://github.com/PaddlePaddle/X2Paddle


