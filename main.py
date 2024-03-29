import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from vit_pytorch.vit_for_small_dataset import ViT
import matplotlib.pyplot as plt
import torch.nn.functional as F

import glob, os, csv
from typing import OrderedDict

# 定义模型
def create_model():
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=45,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=2048,
    )
    state_dict = torch.load("G:\\vision_transformer\\final_model.pth")
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k[:]
        new_sd[name] = v
    model.load_state_dict(new_sd, strict=True)
    model = model.cuda()  # 将模型移动到GPU
    model.eval()  # 设置模型为评估模式
    return model


def predict(image_path, model):
    # 数据预处理
    transform = Compose([
        Resize((224, 224)),  # 添加这一行
        ToTensor(), 
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).cuda()  # 将图像移动到GPU
    with torch.no_grad():
        outputs = model(image)
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class

# 创建模型
model = create_model()

# 对输入图像进行预测
image_path = 'C:\\Users\\苏俊\\Desktop\\images.jpg'
prediction = predict(image_path, model)
print(f'The predicted depression level for the input image is: {prediction}')

