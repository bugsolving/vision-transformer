from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT

import csv

# 基本设置
batch_size = 64 # 一次训练的样本数
epochs = 20 # loop times
lr = 3e-5 # 学习率learning rate
gamma = 0.7 # 更新lr的乘法因子
seed = 42 # seed for random nums


# seed settings
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


device = 'cuda'

# 加载训练目录对应的标签，参数：train_dir训练所用图像所在的文件夹，label_source_dr标签那一堆csv所在的文件夹
def load_labels(train_dir: str, label_source_dir: str):
    pic_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    labels = {}
    for pic in pic_list:
        pic_name = pic.split('/')[-1].split('.')[0]
        pic_number = '_'.join(pic_name.split('_')[:2:])
        label_file = f'{label_source_dir}/{pic_number}_Depression.csv'
        with open(label_file, 'r') as obj:
            reader = csv.reader(obj)
            label_value = 0
            for value in reader:
                label_value = value[0]
                break
            labels.update({pic : int(label_value)})
    print(f'Label length: {len(labels)}')
    return labels


# 加载数据，参数：train_dir训练所用图像所在的文件夹，labels用load_labels返回，应该是字典
def load_data(train_dir: str, test_dir: str, labels):
    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    if test_dir is not None:
        test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    else:
        test_list = []

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    return train_list, test_list, labels

train_list, test_list, labels = load_data('/home/1/resnet/ResNet18/train/Northwind', None, load_labels('/home/1/resnet/ResNet18/train/Northwind', '/home/1/resnet/ResNet18/train'))

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=[labels.get(i) for i in train_list],
                                          random_state=seed)

# 将label字典转换成列表，按train_list的顺序
def label_dict_to_list(train_list):
    return [labels.get(i) for i in train_list]

# 设置ViT
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=45,
    transformer=efficient_transformer,
    channels=3,
).to(device)


# dataset类，给dataloader用
class AVECDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label_list = label_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = self.label_list[idx]

        return img_transformed, label


# image augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

# data loading
train_data = AVECDataset(train_list, label_dict_to_list(train_list), transform=train_transforms)
valid_data = AVECDataset(valid_list, label_dict_to_list(valid_list), transform=test_transforms)
# test_data = AVECDataset(test_list, label_dict_to_list(test_list), transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

print(f'train data loaded: {len(train_data)}, loader: {len(train_loader)}')
print(f'valid data loaded: {len(valid_data)}, loader: {len(valid_loader)}')

# training

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器Adam
optimizer = optim.Adam(model.parameters(), lr=lr)
# 学习率LR的调整类
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# 训练流程
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )