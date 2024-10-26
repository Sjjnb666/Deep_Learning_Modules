import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MaskedAutoencoderVit(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=768,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, mask_prob=0.75):
        super(MaskedAutoencoderVit, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.mask_prob = mask_prob
        self.num_patches = (img_size // patch_size) ** 2

        # 图像块嵌入层 将输入的patch_size**2*3映射到embed_dim的维度
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads,
                                                   dim_feedforward=int(decoder_embed_dim * mlp_ratio), batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # 将编码器输出映射到解码器维度
        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)

        # 重构像素值的输出层 将模型的输出重新映射为图像的大小
        self.output_layer = nn.Linear(decoder_embed_dim, patch_size ** 2 * 3)

    def forward(self, x):
        batch_size = x.size(0)

        # 提取并展平图像块
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # (batch_size.3,num_patches,patch_size,patch_size)
        patches = patches.contiguous().view(batch_size, 3, -1, self.patch_size, self.patch_size)
        # (batch_size,3,num_patches,patch_size,patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.patch_size * self.patch_size * 3)
        # (batch_size,num_patches,patch_size**2*3)

        # 对图像块进行嵌入并加入位置信息
        patches = self.patch_embed(patches)
        # (batch_size,num_patches,embed_dim)
        patches += self.pos_embed
        # (batch_size,num_patches,embed_dim)

        num_masked_patches = int(self.mask_prob * self.num_patches) # 计算需要mask的个数
        mask_indices = torch.randperm(self.num_patches)[:num_masked_patches] # 随机生成被mask的index
        print(mask_indices)
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        mask[mask_indices] = True
        # print(mask.shape)
        masked_patches = patches.clone()
        # torch.arrange(batch_size) 生成一个从 0 到 batch_size-1 的张量，然后使用 unsqueeze(1) 增加一个维度，使其变为 (batch_size, 1) 形状
        masked_patches[torch.arange(batch_size).unsqueeze(1), mask] = 0
        # print(masked_patches.shape)
        # masked_patches (batch_size,num_patches,embed_dim)

        encoded_patches = self.encoder(masked_patches)
        # (batch_size,num_patches,embed_dim)
        encoded_patches = self.enc_to_dec(encoded_patches)
        # (batch_size,num_patches,decoder_embed_dim)

        original_patches = patches.clone()  # 保留原始图像块
        # 将置零的图像块放回原处
        original_patches[torch.arange(batch_size).unsqueeze(1), mask_indices] = masked_patches[
            torch.arange(batch_size).unsqueeze(1), mask_indices]
        # (batch_size,num_patches,embed_dim)

        # 解码器处理
        decoder_patches = self.decoder(tgt=original_patches, memory=encoded_patches)
        # (batch_size,num_patches,decoder_embed_dim)
        reconstruct_patches = self.output_layer(decoder_patches)
        # (batch_size,num_patches,patch_size**2*3)

        # 重构图像块
        reconstructed_patches = reconstruct_patches.view(batch_size, self.num_patches, self.patch_size, self.patch_size, 3)
        # (batch_size,num_patches,patch_size,patch_size,3)
        reconstructed_patches = reconstructed_patches.permute(0, 4, 2, 3, 1).contiguous().view(batch_size, 3,
                                                                                               self.img_size,
                                                                                               self.img_size)
        # (batch_size,3,img_size,img_size)
        return reconstructed_patches

    def train_model(self, train_loader, epochs=10, lr=0.001, device=torch.device('cpu')):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                img, _ = data
                img = img.to(device)
                optimizer.zero_grad()
                reconstructed_img = self(img)
                loss = criterion(reconstructed_img, img)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, test_loader, device=torch.device('cpu')):
        self.eval()
        self.to(device)
        prediction = []
        with torch.no_grad():
            for data in test_loader:
                img, _ = data
                img = img.to(device)
                reconstruct_img = self(img)
                prediction.append(reconstruct_img.cpu())
        return prediction

class SimpleDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, 0

if __name__ == "__main__":
    img_size = 224
    data = torch.randn(4, 3, img_size, img_size)
    train_loader = DataLoader(SimpleDataset(data), batch_size=2, shuffle=True)

    # 实例化模型
    model = MaskedAutoencoderVit(img_size=img_size)

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train_model(train_loader, epochs=5, lr=0.001, device=device)

    # 预测
    predictions = model.predict(train_loader, device=device)
    print(predictions)
