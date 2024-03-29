import os
import csv
import glob
from math import sqrt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from sklearn.model_selection import train_test_split, KFold  # 添加交叉验证
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
import matplotlib.pyplot as plt



# 加载训练目录对应的标签
def load_labels(train_dir: str, label_source_dir: str):
    pic_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    labels = {}
    for pic in pic_list:
        pic_name = os.path.basename(pic).split('.')[0]
        pic_number = '_'.join(pic_name.split('_')[:2:])
        label_file = os.path.join(label_source_dir, f'{pic_number}_Depression.csv')
        with open(label_file, 'r') as obj:
            reader = csv.reader(obj)
            label_value = next(reader)[0]
            labels[pic] = int(label_value)
    print(f'Label length: {len(labels)}')
    return labels

# 加载数据
def load_data(train_dir: str, test_dir: str, labels):
    train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg')) if test_dir else []

    print(f"Train Data: {len(train_list)}")
    print(f"Test Data: {len(test_list)}")

    return train_list, test_list, labels

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[image_path]
        if self.transform:
            image = self.transform(image)
        return image, label

# 使用函数
labels = load_labels('G:\\vision_transformer\\train\\pic', 'G:\\vision_transformer')
train_list, test_list, labels = load_data('G:\\vision_transformer\\train\\pic', None, labels)

# 划分训练集和验证集
train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=[labels.get(i) for i in train_list],
                                          random_state=42)

# 数据预处理
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
])

# 创建数据加载器
train_dataset = CustomDataset(train_list, labels, transform=transform)
valid_dataset = CustomDataset(valid_list, labels, transform=transform)


# 添加交叉验证
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)

# 开始交叉验证
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_list)):
    # 打印当前折数
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 根据折数划分训练集和验证集
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=train_subsampler)
    valid_loader = DataLoader(valid_dataset, batch_size=256, sampler=test_subsampler)





# 定义ViT模型___________________________________________________________________________________________________
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 在模型中添加dropout层
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# 定义损失函数和优化器
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=45,
    dim=512,
    depth=8,
    heads=8,
    mlp_dim=2048,
)
model = model.cuda()  # 将模型移动到GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 80
best_valid_loss = float('inf')
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    for images, labels in train_loader:
        images = images.cuda()  # 将图像移动到GPU
        labels = labels.cuda()  # 将标签移动到GPU

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)  # 记录训练损失
    train_acc = 100. * train_correct / len(train_loader.dataset)

    valid_loss = 0
    valid_correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.cuda()  # 将图像移动到GPU
            labels = labels.cuda()  # 将标签移动到GPU

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            valid_correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_losses.append(valid_loss)  # 记录验证损失
    valid_acc = 100. * valid_correct / len(valid_loader.dataset)

    print ('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Valid Loss: {:.4f}, Valid Acc: {:.2f}%'.format(
        epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))

    torch.save(model.state_dict(), 'best_model.pth')
    print('Model saved to best_model.pth')

# 打印训练损失和验证损失
print('Train Losses:', train_losses)
print('Valid Losses:', valid_losses)


# 可视化训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss")
plt.plot(valid_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
