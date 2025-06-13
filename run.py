import os
import re
from PIL import Image
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

class AnimeExpressionDataset(Dataset):
    def __init__(self, rootDir: str, transform=None):
        self._imagePaths: List[str] = []
        self._labels: List[str] = []
        self._transform = transform         # 模型
        labelSet = set()                    # 标签集合

        for character in os.listdir(rootDir):
            characterDir = os.path.join(rootDir, character)
            if not os.path.isdir(characterDir):
                continue
            for fname in os.listdir(characterDir):
                if not fname.endswith(".png"):
                    continue
                # 1-1-0034-开心.png
                # 获取到 "开心"
                match = re.match(r".+-(.+)\.png", fname)
                if match:
                    expression = match.group(1)
                    self._imagePaths.append(os.path.join(characterDir, fname))
                    self._labels.append(expression)
                    labelSet.add(expression)

        self._labelEncoder = LabelEncoder()
        self._labelEncoder.fit(list(labelSet))
        self._encodedLabels = self._labelEncoder.transform(self._labels)

    def __len__(self) -> int:
        return len(self._imagePaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self._imagePaths[idx]).convert("RGB")
        if self._transform:
            img = self._transform(img)
        label = self._encodedLabels[idx]    # type: ignore
        return img, label                   # type: ignore

    def get_label_encoder(self) -> LabelEncoder:
        return self._labelEncoder


def train(resumeEpoch: int = 0):
    """开始训练

    Args:
        resume_epoch (int, optional): 从上一次 (第 {resume_epoch} 轮) 继续训练. Defaults to 0.

    Raises:
        FileNotFoundError: void
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),         # 缩放
        transforms.CenterCrop(224),     # 中心裁剪 224 * 224
        transforms.ToTensor(),          # 像素值缩放到 [0, 1]
    ])

    dataset = AnimeExpressionDataset("./dataset/mae", transform=transform)
    le = dataset.get_label_encoder()
    numClasses = len(le.classes_)
    print("表情标签类别: ", le.classes_)

    # 划分: 训练集和验证集 (8 : 2)
    trainSize = int(0.8 * len(dataset))
    valSize = len(dataset) - trainSize
    trainSet, valSet = random_split(dataset, [trainSize, valSize])

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=32, shuffle=False)

    # 模型定义
    # 预训练 ResNet18, 修改最后一层适配分类数
    # https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
    # https://doc.embedfire.com/linux/rk356x/Ai/zh/latest/lubancat_ai/base/resnet18_pytorch.html
    # https://blog.csdn.net/m0_64799972/article/details/132753608

    # https://zhuanlan.zhihu.com/p/411530410

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, numClasses)
    model = model.to(device)

    if resumeEpoch > 0:
        # 继续训练, 加载之前的模型参数
        path = f".\\modl\\anime_expression_model_epoch_{resumeEpoch}.pth"
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"已加载模型参数: {path}")
        else:
            raise FileNotFoundError(f"指定的模型文件不存在: {path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    numEpochs = 20 # 总轮数
    trainLosses, valLosses, valAccuracies = [], [], []

    for epoch in range(resumeEpoch, numEpochs):
        model.train()
        runningLoss = 0.0

        # 训练
        for inputs, labels in tqdm(trainLoader, desc=f"Epoch {epoch + 1}/{numEpochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item() * inputs.size(0)

        trainLoss = runningLoss / len(trainLoader.dataset) # type: ignore
        trainLosses.append(trainLoss)

        # 验证
        model.eval()
        valLoss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        valLosses.append(valLoss / len(valLoader.dataset)) # type: ignore
        valAccuracies.append(acc)

        print(f"Epoch {epoch + 1}/{numEpochs}: "
              f"Train Loss = {trainLoss:.4f}, Val Loss = {valLosses[-1]:.4f}, Val Acc = {acc:.4f}")

        # 每轮保存模型
        torch.save(model.state_dict(), f"anime_expression_model_epoch_{epoch + 1}.pth")

if __name__ == "__main__":
    train(resumeEpoch=3)

# ['兴奋' '内疚' '受伤' '吃' '喝醉' '嘲笑' '委屈' '嫌弃' '害羞' '尴尬' '工作' '开心' '得意' '忍耐' '怀疑'
#  '思考' '悲伤' '惊吓' '惊恐' '愤怒' '慌张' '憋笑' '担心' '放松' '无语' '疑惑' '疲惫' '痛苦' '着急'
#  '睡觉' '紧张' '观察' '认真' '震惊' '饥饿']