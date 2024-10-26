import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import GPU

class FCN(nn.Module):
    def __init__(self,n_classes):
        """
        初始化
        :param n_classes: 分割输出的类别数量
        """
        super(FCN,self).__init__()

        # VGG16
        # 普通卷积 (Input+2*padding-kernel)/stride+1
        self.features = nn.Sequential(
            # 输出 (3,224,224)
            # conv1
            nn.Conv2d(3,64,kernel_size=3,padding=1), # (64,224,224)
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1), # (64,224,224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # (64,112,112)

            # conv2
            nn.Conv2d(64,128,kernel_size=3,padding=1), # (128,112,112)
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1), # (128,112,112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # (128,56,56)

            # conv3
            nn.Conv2d(128,256,kernel_size=3,padding=1), # (256,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1), # (256,56,56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1), # (256,56,56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # (256,28,8)

            # conv4
            nn.Conv2d(256,512,kernel_size=3,padding=1), # (512,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,28,28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,28,28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # (512,14,14)

            # conv5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # (512,14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # (512,7,7)

        )
        # 全卷积层
        self.fc6 = nn.Conv2d(512,4096,kernel_size=7) # (4096,1,1)
        self.fc7 = nn.Conv2d(4096,4096,kernel_size=1) # (4096,1,1)
        self.score_fr = nn.Conv2d(4096,n_classes,kernel_size=1) # (n_classes,1,1)

        # 转置卷积层 (上采样) out_size = (in_size - 1) * S + K - 2P = （input-1）*stride+k-2
        self.upscore2 = nn.ConvTranspose2d(n_classes,n_classes,kernel_size=4,stride=2,padding=1,bias=False)  # 上采样倍数为2
        self.upscore8 = nn.ConvTranspose2d(n_classes,n_classes,kernel_size=16,stride=8,padding=4,bias=False) # 上采样倍数为8

        # 1x1卷积用于减少通道数
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=1)  # 输入： (256, 28, 28)，输出： (n_classes, 28, 28)
        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=1)  # 输入： (512, 14, 14)，输出： (n_classes, 14, 14)

    def forward(self,x):
        """
        网络的前向传播
        :param x: 输入张量
        :return: 输出分割图
        """
        pool3 = self.features[0:17](x) # (256,28,28)
        pool4 = self.features[17:24](pool3) # (512,14,14)
        pool5 = self.features[24:](pool4) # (512,7,7)

        fc6 = F.relu(self.fc6(pool5)) # (4096,1,1)
        fc7 = F.relu(self.fc7(fc6)) # (4096,1,1)
        score_fr = self.score_fr(fc7) # (n_classes,1,1)

        # 上采样 倍率为2
        upscore2 = self.upscore2(score_fr) # (n_classes,2,2)

        # 剪裁pool4以匹配upscore2的尺寸
        score_pool4c = self.crop(pool4,upscore2)
        score_pool4c = self.score_pool4(score_pool4c) # (n_classes,2,2)

        fuse_pool4 = upscore2 + score_pool4c # (n_classes,2,2)

        # 继续上采样两倍
        upscore_pool4 = self.upscore2(fuse_pool4) # (n_classes,4,4)

        # 剪裁pool3匹配尺寸
        score_pool3c = self.crop(pool3,upscore_pool4)
        score_pool3c = self.score_pool3(score_pool3c) # (n_classes,4,4)

        fuse_pool3 = upscore_pool4 + score_pool3c # (n_classes,4,4)

        upscore8 = self.upscore8(fuse_pool3) # (n_classes,32,32)
        out = F.interpolate(upscore8, size=x.size()[2:], mode='bilinear', align_corners=False) # (n_classes,H,W)
        return out

    def crop(self,target,refer):
        """
        剪裁
        :param target: 需要剪裁的尺寸 (B,C,H,W)
        :param refer: 参考尺寸 (B,C,H,W)
        :return: 剪裁后的张量
        """
        target_size = target.size()[2:] # (H,W)
        refer_size = refer.size()[2:] # (H,W)
        start = [(target_size[i] - refer_size[i]) // 2 for i in range(2)] # 寻找H和W的中点
        end = [start[i] + refer_size[i] for i in range(2)]
        return target[:, :, start[0]:end[0], start[1]:end[1]]

    def train_model(self,dataloader,loss_fn,optimizer,num_epochs=1):
        """
        训练模型
        :param dataloader: 训练数据加载器
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param num_epochs: 训练轮数
        :return: None
        """
        for epoch in range(num_epochs):
            self.train()
            all_loss = 0.0
            for inputs,targets in dataloader:
                inputs,targets = inputs.to(GPU.get_device()),targets.to(GPU.get_device())
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                targets = targets.squeeze(1).long() # (B,C,H,W)->(B,H,W)
                outputs = outputs.squeeze(1)
                # outputs = F.interpolate(outputs,size=targets.size()[1:],mode='bilinear',align_corners=False)
                loss = loss_fn(outputs,targets)
                loss.backward()
                optimizer.step()
                all_loss += loss.item()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {all_loss / len(dataloader)}')

    def predict(self, dataloader):
        """
        使用训练好的模型进行预测并可视化。

        参数：
        - dataloader (DataLoader): 测试数据的加载器。
        """
        self.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(GPU.get_device())
                print(inputs.shape)
                outputs = self.forward(inputs)
                print(outputs.shape)
                _, pred = torch.max(outputs, 1)
                # visualize(inputs[0], targets[0], outputs[0], idx)
        return pred


def visualize(image, target, output, idx):
    image = image.squeeze().permute(1, 2, 0).cpu().numpy() # (C,H,W)
    target = target.squeeze().cpu().numpy()
    output = output.squeeze().detach().cpu().numpy().argmax(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[1].imshow(target)
    axes[1].set_title('Target Mask')
    axes[2].imshow(output)
    axes[2].set_title('Predicted Mask')
    plt.show()


# 主函数，用于训练和测试
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN(n_classes=21).to(device)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # 下载VOC数据集
    dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=False, transform=transform,
                              target_transform=transform)

    # 取一个子集用于训练
    subset_indices = list(range(100))  # 取前100个样本
    subset = Subset(dataset, subset_indices)

    # 创建数据加载器
    dataloader = DataLoader(subset, batch_size=1, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train_model(dataloader, criterion, optimizer, num_epochs=5)

    # 预测并可视化结果
    pred = model.predict(dataloader)
    print(pred.shape)


if __name__ == "__main__":
    main()
