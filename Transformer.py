import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader
import GPU


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        初始化位置编码
        :param d_model: 词嵌入的维度
        :param max_seq_length: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵
        self.encoding = torch.zeros(max_seq_length, d_model)
        self.encoding.requires_grad = False
        # 生成位置矩阵, unsqueeze使得维度为(max_seq_length,1)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 正余弦部分
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (batch_size,seq_len,d_model)
        :return: 加上位置编码的张量
        """
        batch_size, seq_len, d_model = x.size()
        return x + self.encoding[:seq_len, :].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        """
        初始化多头注意力机制
        :param d_model: 词嵌入的维度
        :param nhead: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        # 确保 d_model 可以被 nhead 整除
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 定义线性变换层
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def attention(self, query, key, value, mask=None):
        """
        计算注意力分数
        :param query: 查询张量，(batch_size,nhead,seq_len,d_k)
        :param key: 键张量，(batch_size,nhead,seq_len,d_k)
        :param value: 值张量，(batch_size,nhead,seq_len,d_k)
        :param mask: 掩码张量，(batch_size,1,seq_len)
        :return: 和value乘好的和注意力权重张量
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)

        # 计算注意力权重
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数。

        参数：
        query: 查询张量，形状为 (batch_size, seq_len, d_model)。
        key: 键张量，形状为 (batch_size, seq_len, d_model)。
        value : 值张量，形状为 (batch_size, seq_len, d_model)。
        mask : 掩码张量，形状为 (batch_size, 1, seq_len)。

        返回：
        torch.Tensor: 经过多头注意力机制后的张量，形状为 (batch_size, seq_len, d_model)。
        """
        batch_size = query.size(0)

        # 线性变换并分头
        query = self.query(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        x, attn = self.attention(query, key, value, mask=mask)

        # 组合多头结果
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, *args, **kwargs):
        """

        :param d_model:
        :param dim_feedforward:
        :param dropout:
        """
        super().__init__(*args, **kwargs)
        # 定义两层全连接网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (batch_size,seq_len,d_model)
        :return: 经过前馈神经网络后的张量
        """
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        初始化编码器
        :param d_model: 词嵌入的维度
        :param nhead: 注意力头的个数
        :param dim_feedforward: 前馈神经网络的隐藏层维度
        :param dropout: 概率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        前向传播
        :param src: 源输入(batch_size,seq_len,d_model)
        :param src_mask: 源序列掩码(batch_size,1,seq_len)
        :return: 编码层后的张量
        """
        # 多头注意力机制
        src2 = self.self_attn(src, src, src, src_mask)
        # 残差连接和归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈网络
        src2 = self.feed_forward(src)
        # 残差连接和归一化
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        初始化解码层
        :param d_model: 词嵌入维度
        :param nhead: 注意力头的个数
        :param dim_feedforward: 前馈神经网络的隐藏层维度
        :param dropout: 概率
        """
        super(DecoderLayer, self).__init__()
        # 自注意力机制
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # 连接编码器输出
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        # 定义层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 定义 dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        前向传播函数
        :param tgt: 目标序列输入 (batch_size,seq_len,d_model)
        :param memory: 编码器输出 (batch_size,src_seq_len,d_model)
        :param tgt_mask: 目标序列掩码 (batch_size,seq_len,seq_len)
        :param memory_mask: 编码器输出掩码 (batch_size,1,src_seq_len)
        :return: 经过解码器层后的张量
        """
        # 自注意力机制
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # 残差连接和层归一化
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 交叉注意力机制
        tgt2 = self.multihead_attn(tgt, memory, memory, memory_mask)
        # 残差连接和层归一化
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 前馈神经网络
        tgt2 = self.feed_forward(tgt)
        # 残差连接和层归一化
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class BaseTransformer(nn.Module):
    def __init__(self, d_model, nhead,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length,
                 vocab_size, dropout=0.1):
        """
        初始化Transformer模型。

        参数：
        d_model : 词嵌入的维度（即特征数）。
        nhead: 多头注意力机制中的头数。
        num_encoder_layers : 编码器的层数。
        num_decoder_layers : 解码器的层数。
        dim_feedforward : 前馈网络中的隐藏层维度。
        max_seq_length : 序列的最大长度。
        vocab_size : 词汇表的大小。
        dropout: dropout概率。
        """
        super(BaseTransformer, self).__init__()
        self.d_model = d_model
        # 嵌入层
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        # 编码器
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        # 解码器
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        # 输出层，将解码器的输出映射到词汇表大小
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播函数。

        参数：
        src : 源序列输入，形状为 (batch_size, src_seq_length)。
        tgt : 目标序列输入，形状为 (batch_size, tgt_seq_length)。
        src_mask : 源序列的掩码，形状为 (batch_size, 1, src_seq_length)。
        tgt_mask : 目标序列的掩码，形状为 (batch_size, tgt_seq_length, tgt_seq_length)。

        返回：
        模型输出，形状为 (batch_size, tgt_seq_length, vocab_size)。
        """
        # 获取源序列的嵌入表示并且进行位置编码
        src_embedding = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_embedding = self.position_encoding(src_embedding)  # 获取目标序列的嵌入表示并进行位置编码
        tgt_embedding = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedding = self.position_encoding(tgt_embedding)
        # 编码器部分
        memory = src_embedding
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
        # 解码器部分
        output = tgt_embedding
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, src_mask)
        # 最后输出层
        output = self.fc_out(output)

        return output

    def make_src_mask(self, src):
        """
        指示序列中哪些位置是填充
        :param src: 源输入序列 (batch_size,src_seq_length)
        :return: 源序列的掩码 (batch_size,1,1,src_seq_length)
        """
        scr_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return scr_mask

    def make_future_mask(self, tgt):
        """
        生成一个上三角矩阵，用于屏蔽解码器在未来的输入
        :param tgt: 目标序列
        :return: 上三角矩阵 (tgt_seq_length,tgt_seq_length)
        """
        tgt_seq_length = tgt.size(1)
        future_mask = torch.triu(torch.ones((tgt_seq_length, tgt_seq_length)), diagonal=1).to(tgt.device)
        # 把矩阵里面是1的替换成负无穷，是0的替换成0
        return future_mask.masked_fill(future_mask == 1, float('-inf')).masked_fill(future_mask == 0, float(0.0))

    def train_step(self, optimizer, loss_fn, src, tgt, src_mask, tgt_mask):
        """
        单步训练过程
        :param optimizer: 优化器
        :param loss_fn: 损失函数
        :param src: 源输入序列 (batch_size,src_seq_length)
        :param tgt: 目标序列输入 (batch_size,tgt_seq_length)
        :param src_mask: 源序列掩码 (batch_size,1,src_seq_length)
        :param tgt_mask: 目标序列掩码 (batch_size,tgt_seq_length,tgt_seq_length)
        :return: 当前训练步骤的损失
        """
        self.train()  # 训练模式

        # 前向传播
        output = self(src, tgt, src_mask, tgt_mask)
        output = output.view(-1, output.size(-1))
        tgt = tgt.view(-1)

        # 计算损失
        loss = loss_fn(output, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_model(self, train_loader, optimizer, loss_fn, epochs):
        """
        模型训练过程
        :param train_loader: 训练数据Dataloader
        :param optimizer: 优化器
        :param loss_fn: 损失函数
        :param epochs: 训练轮速
        :return: None
        """
        device = GPU.get_device()
        self.to(device)
        for epoch in range(epochs):
            epoch_loss = 0
            for src, tgt in train_loader:
                src_mask = self.make_src_mask(src).to(device)
                tgt_mask = self.make_future_mask(tgt).to(device)
                src, tgt = src.to(device), tgt.to(device)

                loss = self.train_step(optimizer, loss_fn, src, tgt, src_mask, tgt_mask)
                epoch_loss += loss

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')

    def predict(self, src, max_length):
        """
        模型预测过程
        :param src: 源序列输入 (batch_size,src_seq_length)
        :param max_length: 生成序列的最大长度
        :return: 生成的目标序列(batch_size,max_length)
        """
        device = GPU.get_device()
        self.to(device)
        self.eval()
        src_mask = self.make_src_mask(src).to(device)
        src = src.to(device)
        # 初始化目标序列
        tgt = torch.zeros((src.size(0), 1), dtype=torch.long, device=src.device)
        generated = tgt

        for _ in range(max_length):
            tgt_mask = self.make_future_mask(generated).to(device)
            output = self(src, generated, src_mask, tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch_size,vocab_size)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == 0:  # 假设0是结束标志
                break

        return generated[:, 1:]


# 示例用法
# d_model = 512
# nhead = 8
# num_encoder_layers = 6
# num_decoder_layers = 6
# dim_feedforward = 2048
# max_seq_length = 100
# vocab_size = 10000
# dropout = 0.1
#
# model = BaseTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length,
#                         vocab_size, dropout)
#
# # 随机生成源序列和目标序列
# src = torch.randint(0, vocab_size, (32, max_seq_length))  # (batch_size, src_seq_length)
# tgt = torch.randint(0, vocab_size, (32, max_seq_length))  # (batch_size, tgt_seq_length)
#
# # 前向传播
# output = model(src, tgt)
# print(output.shape)  # 输出形状: (batch_size, tgt_seq_length, vocab_size)
# 示例用法
def main():
    # 参数设置
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 100
    vocab_size = 10000
    dropout = 0.1
    epochs = 10

    # 数据加载
    # 假设 train_loader 是 DataLoader 的实例，已经定义和加载数据
    # 这里仅用随机数据作为示例
    train_data = [(torch.randint(0, vocab_size, (10,)), torch.randint(0, vocab_size, (10,))) for _ in range(100)]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 模型实例化
    model = BaseTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length,
                            vocab_size, dropout)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 假设 0 是填充标志

    # 训练模型
    model.train_model(train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs)

    # 示例预测
    src = torch.randint(0, vocab_size, (1, 10))  # 随机输入数据
    generated_seq = model.predict(src, max_length=10)
    print("Generated sequence:", generated_seq)


if __name__ == "__main__":
    main()
