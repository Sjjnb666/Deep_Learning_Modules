import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,ViTModel,ViTFeatureExtractor,BertTokenizer
from PIL import Image
import requests
from io import BytesIO
from torch.utils.data import DataLoader,Dataset

class CLIP(nn.Module):
    def __init__(self,text_model_name='bert-base-uncased',image_model_name='google/vit-base-patch16-224',embed_dim=512):
        """
        初始化CLIP
        :param text_model_name:文本编码器的名字
        :param image_model_name:图片编码器的名字
        :param embed_dim:嵌入维度
        文本嵌入 (batch_size,embed_dim)
        图像嵌入 (batch_size,embed_dim)
        """
        super(CLIP,self).__init__()
        # 初始化文本编码器
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size,embed_dim) # (batch_size,hidden_size)->(batch_size,embed_dim)

        # 初始化图片编码器
        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size,embed_dim)# (batch_size,hidden_size)->(batch_size,embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def encode_text(self,input_ids,attention_mask):
        """
        文本编码
        :param input_ids: 文本输入 (batch_size,seq_len)
        :param attention_mask: 文本注意力掩码 (batch_size,seq_len)
        :return: 文本嵌入 (batch_size,embed_dim)
        """
        outputs = self.text_encoder(input_ids=input_ids,attention_mask=attention_mask) # (batch_size,seq_len,hidden_size)
        text_features = self.text_proj(outputs.last_hidden_state[:,0,:]) # (batch_size,hidden_size)->(batch_size,embed_dim)
        return text_features

    def encode_image(self,pixel_values):
        """
        编码图像
        :param pixel_values: 图像输入(batch_size,3,height,weight)
        :return: 图像嵌入 (batch_size,embed_dim)
        """
        outputs = self.image_encoder(pixel_values=pixel_values) # (batch_size,seq_len,hidden_size)
        image_features = self.image_proj(outputs.last_hidden_state[:,0,:]) # (batch_size,hidden_size)->(batch_size,embed_dim)
        return image_features

    def forward(self,input_ids,attention_mask,pixel_values):
        """
        前向传播
        :param input_ids: 文本输入 (batch_size,seq_len)
        :param attention_mask: 自注意力掩码 (batch_size,seq_len)
        :param pixel_values: 图像输入 (batch_size,3,weight,height)
        :return:
        logits_per_text :图像和文本之间的相似度 (batch_size,batch_size)
        logits_per_image: 图像和文本之间的相似度 (batch_size,batch_size)
        """
        text_features = self.encode_text(input_ids,attention_mask) # (batch_size,embed_dim)
        image_features = self.encode_image(pixel_values)

        # L2正则化
        text_features = F.normalize(text_features,dim=-1)
        image_features = F.normalize(image_features,dim=-1)

        # 计算相似度
        logit_scale = self.logit_scale.exp()
        # (batch_size, embed_dim) @ (embed_dim, batch_size) -> (batch_size, batch_size)
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        logits_per_image = logits_per_text.t()  # (batch_size, batch_size)

        return logits_per_text,logits_per_image

    def get_loss(self,logits_per_text,logits_per_image):
        """
        计算对比损失
        :param logits_per_text:文本相似度 (batch_size,batch_size)
        :param logits_per_image:图像相似度 (batch_size,batch_size)
        :return:对比损失
        """
        labels = torch.arange(logits_per_text.size(0)).to(logits_per_text.device) # (batch_size)
        loss_text = F.cross_entropy(logits_per_text,labels) # (1)
        loss_image = F.cross_entropy(logits_per_image,labels) # (1)
        loss = (loss_image+loss_text)/2
        return loss

    def predict(self, pixel_values, labels):
        self.eval()
        with torch.no_grad():
            # 编码所有标签文本
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            texts = [f"A photo of a {label}." for label in labels]
            inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()

            text_features = self.encode_text(input_ids, attention_mask)  # (num_labels, embed_dim)
            text_features = F.normalize(text_features, dim=-1)

            # 编码输入图像
            pixel_values = pixel_values.cuda()
            image_features = self.encode_image(pixel_values)  # (batch_size, embed_dim)
            image_features = F.normalize(image_features, dim=-1)

            # 计算相似度
            logit_scale = self.logit_scale.exp()
            logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale  # (batch_size, num_labels)
            similarity = F.softmax(logits_per_image, dim=-1)  # (batch_size, num_labels)

        return similarity
# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, texts, images, tokenizer, feature_extractor):
        self.texts = texts  # 文本列表
        self.images = images  # 图像列表
        self.tokenizer = tokenizer  # 文本编码器
        self.feature_extractor = feature_extractor  # 图像编码器

    def __len__(self):
        return len(self.texts)  # 数据集的大小

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        # inputs['input_ids'] 维度: (1, sequence_length)
        # inputs['attention_mask'] 维度: (1, sequence_length)
        pixel_values = self.feature_extractor(images=image, return_tensors="pt")['pixel_values']
        # pixel_values 维度: (1, 3, height, width)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), pixel_values.squeeze(0)
        # 返回: input_ids (sequence_length), attention_mask (sequence_length), pixel_values (3, height, width)

# 示例: 初始化 CLIP 模型
model = CLIP().cuda()

# 示例: 输入数据
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# 获取示例图片
urls = ["https://p2.ssl.qhimgs1.com/t01d16656805eba9022.jpg", "https://p2.ssl.qhimgs1.com/t01d16656805eba9022.jpg"]
images = [Image.open(BytesIO(requests.get(url).content)) for url in urls]
texts = ["a photo of a cat", "a photo of a dog"]

# 创建数据集和数据加载器
dataset = CustomDataset(texts, images, tokenizer, feature_extractor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 训练模型
model.train()
for epoch in range(5):
    for input_ids, attention_mask, pixel_values in dataloader:
        input_ids = input_ids.cuda()  # (batch_size, sequence_length)
        attention_mask = attention_mask.cuda()  # (batch_size, sequence_length)
        pixel_values = pixel_values.cuda()  # (batch_size, 3, height, width)

        # 前向传播
        logits_per_text, logits_per_image = model(input_ids, attention_mask, pixel_values)
        # logits_per_text 维度: (batch_size, batch_size)
        # logits_per_image 维度: (batch_size, batch_size)

        # 计算损失
        loss = model.get_loss(logits_per_text, logits_per_image)  # (1)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")