import os
from PIL import Image
from tqdm import tqdm

inputRoot = "./dataset/mae"
outputRoot = "./dataset/processed"
targetSize = (224, 224)

os.makedirs(outputRoot, exist_ok=True)

for character in os.listdir(inputRoot):
    inputDir = os.path.join(inputRoot, character)
    outputDir = os.path.join(outputRoot, character)
    os.makedirs(outputDir, exist_ok=True)

    for fname in tqdm(os.listdir(inputDir), desc=f"Resizing {character}"):
        if not fname.endswith(".png"):
            continue
        inputPath = os.path.join(inputDir, fname)
        outputPath = os.path.join(outputDir, fname)

        try:
            img = Image.open(inputPath).convert("RGB")
            img = img.resize(targetSize, Image.BICUBIC) # type: ignore
            img.save(outputPath)
        except Exception as e:
            print(f"处理 {inputPath} 失败：{e}")
