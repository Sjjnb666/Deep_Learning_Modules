import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import transforms, datasets
import GPU
import os

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        """
        :param in_channels: 输入图像通道数
        :param patch_size:每个patch的大小
        :param emb_size: 嵌入维度大小
        :param img_size: 输入图像的尺寸
        """
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 利用卷积层把每个patch投射为token
        self.flatten = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        前向传播
        transformer要求的输入是[seq_len,hidden_size]
        :param x: 输入的图片 [batch_size,channels,H,W]
        :return: 输出的对应token
        """
        x = self.flatten(x)  # [batch_size,embed_size,n_patches_sqrt,n_patches_sqrt]
        x = x.flatten(2)  # [batch_size,embed_size,n_patches]
        x = x.transpose(1, 2)  # [batch_size,n_patches,embed_size]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0.):
        """
        :param emb_size: 嵌入维度大小
        :param num_heads: 头数量
        :param dropout: 概率
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        assert (
                self.head_dim * num_heads == emb_size
        ), "Embedding size needs to be divisible by num_heads"

        # 定义KQV线性层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, values, keys, query, mask=None):
        """
        :param values: V
        :param keys: K
        :param query: Q
        :param mask: 掩码，忽略某部分的值
        :return: None
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 塑性为(batch_size,num_patches,num_heads,head_dim)
        values = values.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        queries = query.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        # 计算注意力权重
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, values)
        x = x.transpose(1, 2).contiguous().view(N, query_len, self.emb_size)
        return self.fc_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0., forward_expansion=4):
        """
        :param emb_size: 嵌入层维度
        :param num_heads: 头的数目
        :param dropout: 概率
        :param forward_expansion: 前馈神经网络中间层的维度拓展
        """
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        """
        :param value: V
        :param key: K
        :param query: Q
        :param mask: 掩码，用于忽略某些位置的值
        :return:
        """
        attention = self.attention(value, key, query, mask)

        # Add+Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Vit(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 in_channels=3, num_classes=1000, emb_size=768,
                 depth=12, num_heads=8, dropout=0., forward_expansion=4):
        """
        :param img_size: 输入图像的尺寸
        :param patch_size: 每个patch的尺寸
        :param in_channels: 输入图像的通道数
        :param num_classes: 分类数量
        :param emb_size: 嵌入维度大小
        :param depth: Transformer块的数量
        :param num_heads: 多头注意力机制中的头数量
        :param dropout: Dropout概率
        :param forward_expansion: 前馈网络中间层的扩展倍数
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # 构建位置编码 [1,1+n_patches,emb_size] 这个维度为了方便和后面相加
        self.position_encoding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dropout, forward_expansion) for _ in range(depth)]
        )

        # 恒等变换层，后面forward需要用
        self.to_cls_token = nn.Identity()

        # 最后的分类
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        """
        :param x: 输入图片
        :return:
        """
        x = self.patch_embedding(x)  # [batch_size,n_patches,emb_size]
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size,1,emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size,1+n_patches,emb_size]
        x += self.position_encoding[:, :(x.size(1))]
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, x, x, mask=None)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

    def train_model(self, train_loader, val_loader, epochs, lr, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.to(device)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(total_loss)

            avg_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_loader, device)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    def evaluate(self, val_loader, device):
        """
        :param val_loader: 测试数据
        :param device: 设备
        :return:
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct/total

    def predict(self, image, device):
        self.eval()
        image = image.to(device)
        with torch.no_grad():
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
        return predicted


from torch.utils.data import Subset
import numpy as np
# 超参数设置
img_size = 256
patch_size = 16
in_channels = 3
num_classes = 10
emb_size = 768
depth = 12
num_heads = 8
dropout = 0.1
forward_expansion = 4
epochs = 5
learning_rate = 3e-4
batch_size = 32
subset_size = 1000  # 子集大小

# 获取设备
device = GPU.get_device()

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载完整的训练和验证数据集
full_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
full_val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 创建训练和验证子集
train_indices = np.random.choice(len(full_train_dataset), subset_size, replace=False)
val_indices = np.random.choice(len(full_val_dataset), subset_size // 10, replace=False)

train_subset = Subset(full_train_dataset, train_indices)
val_subset = Subset(full_val_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# 创建模型实例
model = Vit(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    emb_size=emb_size,
    depth=depth,
    num_heads=num_heads,
    dropout=dropout,
    forward_expansion=forward_expansion
)

# 训练模型
model.train_model(train_loader, val_loader, epochs, learning_rate, device)

# 预测示例
sample_image, _ = val_subset[0]
sample_image = sample_image.unsqueeze(0)  # 添加batch维度
predicted_label = model.predict(sample_image, device)
print(f'Predicted Label: {predicted_label.item()}')