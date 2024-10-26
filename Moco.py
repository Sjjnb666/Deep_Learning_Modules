import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader, TensorDataset


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999):
        """
        初始化MoCo模型
        :param base_encoder: 基础编码器模型，这里用的是resnet
        :param dim: 特征向量的维度，默认为128
        :param K: 队列大小，默认大小为65536
        :param m: 动量更新参数，默认值为0.999
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        # 初始化查询编码器和键编码器
        self.query_encoder = base_encoder(num_classes=dim)  # (batch_size,dim)
        self.key_encoder = base_encoder(num_classes=dim)  # (batch_size,dim)

        # key编码器的参数直接复制query编码器，key编码器不进行梯度更新
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # 不更新梯度

        # 初始化队列，用于存储键的特征向量
        self.register_buffer("queue", torch.randn(dim, K))  # (dim,K) 创建一个名字叫queue的buffer缓冲
        self.queue = nn.functional.normalize(self.queue, dim=0)  # 归一化队列

        # 初始化队列指针
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        """
        动量的形式更新键编码器的参数
        """
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q * (1 - self.m)  # 动量更新的办法

    @torch.no_grad()
    def queue_update(self, keys):
        """
        更新队列，新的键向量加入队列并且移除老的键向量
        :param keys: 键向量 (batch_size,dim)
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # 更新队列
        self.queue[:, ptr:ptr + batch_size] = keys.T  # 插入新键
        ptr = (ptr + batch_size) % self.K  # 更新队列指针的位置
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        前向传播
        :param im_q: 查询图像 [batch_size,3,H,W]
        :param im_k: 键图像 [batch_size,3,H,W]
        """
        q = self.query_encoder(im_q)  # (batch_size,dim)
        with torch.no_grad():
            self.momentum_update()
            k = self.key_encoder(im_k)  # (batch_size,dim)

        # nc,nc->n的意思是消除第二个维度，输出的只有一个维度也就是batch_size
        loss_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (batch_size,1)
        loss_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # 维度为[batch_size, K]

        logits = torch.cat([loss_pos,loss_neg],dim=1) # (batch_size,1+K)
        logits = logits/0.07 # 温度参数

        labels = torch.zeros(logits.shape[0],dtype=torch.long).to(logits.device) # (batch_size)

        loss = nn.CrossEntropyLoss()(logits,labels) # 计算损失
        self.queue_update(k)
        return loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MoCo(resnet50).to(device)  # 模型移至设备
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器，学习率为0.001


train_data = torch.randn(100, 3, 224, 224).to(device)  # 随机数据，维度为[100, 3, 224, 224]
train_labels = torch.randint(0, 10, (100,)).to(device)  # 随机标签，维度为[100]
train_dataset = TensorDataset(train_data, train_labels)  # 创建数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 数据加载器

def train(model, data_loader, optimizer, epochs=10):
    """
    训练函数。
    :param model: 训练的模型
    :param data_loader: 数据加载器
    :param optimizer: 优化器
    :param epochs: 训练的总轮数
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for imgs, _ in data_loader:
            imgs = imgs.to(device)  # 将图片移至设备
            im_q = imgs  # 查询图片
            im_k = imgs.clone()  # 键图片，此处简化为复制查询图片
            loss = model(im_q, im_k)  # 计算损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累计损失

        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {total_loss / len(data_loader):.4f}")  # 打印平均损失

train(model, train_loader, optimizer, epochs=10)  # 开始训练
