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
    def __init__(self, root_dir: str, transform=None):
        self.image_paths: List[str] = []
        self.labels: List[str] = []
        self.transform = transform
        label_set = set()

        for character in os.listdir(root_dir):
            character_dir = os.path.join(root_dir, character)
            if not os.path.isdir(character_dir):
                continue
            for fname in os.listdir(character_dir):
                if not fname.endswith(".png"):
                    continue
                match = re.match(r".+-(.+)\.png", fname)
                if match:
                    expression = match.group(1)
                    self.image_paths.append(os.path.join(character_dir, fname))
                    self.labels.append(expression)
                    label_set.add(expression)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(label_set))
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.encoded_labels[idx] # type: ignore
        return img, label # type: ignore

    def get_label_encoder(self) -> LabelEncoder:
        return self.label_encoder


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = AnimeExpressionDataset("./dataset/mae", transform=transform)
    le = dataset.get_label_encoder()
    num_classes = len(le.classes_)
    print("表情标签类别：", le.classes_)

    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 模型定义
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        val_losses.append(val_loss / len(val_loader.dataset))
        val_accuracies.append(acc)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Val Loss = {val_losses[-1]:.4f}, Val Acc = {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "anime_expression_model.pth")
    print("模型保存为 anime_expression_model.pth")

    # 可视化训练过程
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig("training_plot.png")
    print("训练过程图已保存为 training_plot.png")


if __name__ == "__main__":
    train()
