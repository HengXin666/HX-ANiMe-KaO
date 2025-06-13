import os
import re
from typing import List

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from sklearn.preprocessing import LabelEncoder

modDir = ".\\modl"

# 载入标签编码器
LABELS = ['兴奋', '内疚', '受伤', '吃', 
          '喝醉', '嘲笑', '委屈', '嫌弃', 
          '害羞', '尴尬', '工作', '开心', 
          '得意', '忍耐', '怀疑', '思考', 
          '悲伤', '惊吓', '惊恐', '愤怒', 
          '慌张', '憋笑', '担心', '放松', 
          '无语', '疑惑', '疲惫', '痛苦', 
          '着急', '睡觉', '紧张', '观察', 
          '认真', '震惊', '饥饿']

labelEncoder = LabelEncoder()
labelEncoder.fit(LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 获取模型列表
def getListModels() -> List[str]:
    return sorted([
        f for f in os.listdir(modDir) if re.match(r"anime_expression_model_epoch_\d+\.pth", f)
    ])

# 加载模型
def loadModel(modelPath: str, numClasses: int) -> nn.Module:
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, numClasses)
    model.load_state_dict(torch.load(f"{modDir}\\{modelPath}", map_location=device))
    model.to(device)
    model.eval()
    return model

# 预测函数
def predict(img: Image.Image, modelName: str) -> str:
    imgTensor = transform(img).unsqueeze(0).to(device) # type: ignore

    model = loadModel(modelName, numClasses=len(labelEncoder.classes_))
    with torch.no_grad():
        outputs = model(imgTensor)
        _, pred = torch.max(outputs, 1)
        predLabel = labelEncoder.inverse_transform(pred.cpu().numpy())[0]
    return f"识别表情：{predLabel}"

# Gradio 接口
modelChoices = getListModels()

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="上传动漫角色图像"),
        gr.Dropdown(choices=modelChoices, label="选择模型", 
                    value=modelChoices[-1] if modelChoices else None),
    ],
    outputs=gr.Textbox(label="预测结果"),
    title="二次元表情识别系统",
    description="上传动漫角色头像, 自动识别其表情~"
)

if __name__ == "__main__":
    demo.launch()
