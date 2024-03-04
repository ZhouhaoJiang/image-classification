import torch
import os
import torch.nn as nn
from torch import save
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torchvision.models as models
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# import os
# os.environ["OMP_NUM_THREADS"] = "1"

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

writer = SummaryWriter('runs/fashion_mnist_experiment')


# 指定数据集的路径
data_dir = 'dataset'

# 初始化数据集和数据加载器
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=os.cpu_count())
#                for x in ['train', 'val']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 设置设备为CUDA（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')

# 初始化模型
# model = models.mobilenet_v2(pretrained=True)
# model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
# model.classifier[1] = nn.Linear(model.last_channel, 5)  # 修改最后一层以适应二分类任务
# 从已有模型加载
# model = models.mobilenet_v2(pretrained=False)
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, 4)  # 修改最后一层以适应二分类任务
# model.load_state_dict(torch.load('result/four_class.pth', map_location='cpu'))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 如果有保存的最佳模型权重，可以通过以下变量追踪
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())


def main():
    global best_acc
    global best_model_wts

    num_epochs = 10000
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm创建进度条
            pbar = tqdm(dataloaders[phase], total=len(dataloaders[phase]),
                        desc=f'Epoch {epoch + 1}/{num_epochs} Phase: {phase}')

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条的描述
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=running_corrects.double() / (pbar.n * 32))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'Epoch {epoch}/{num_epochs - 1} | Phase: {phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if phase == 'val':
                try:
                    writer.add_scalar('Loss/val', running_loss, epoch)
                    writer.add_scalar('Accuracy/val',running_corrects,epoch)
                except Exception as e:
                    print(e)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save(model.state_dict(), './model/best_model.pth')
                if (epoch + 1) % 5 == 0:
                    save(model.state_dict(), f'./model/epoch_{epoch + 1}_model.pth')
            else:
                try:
                    writer.add_scalar('Loss/train', running_loss, epoch)
                    writer.add_scalar('Accuracy/train',running_corrects,epoch)
                except Exception as e:
                    print(e)



    print('Training complete')

    model.load_state_dict(best_model_wts)
    model.eval()
    running_corrects = 0

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / dataset_sizes['val']
    print(f'Test Accuracy: {accuracy:.4f}')
    writer.close()


if __name__ == '__main__':
    main()

