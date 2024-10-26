import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import math


class BertConfig:
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
                 max_pos_embedding=512, type_vocab_size=2, init_range=0.02,
                 layer_norm_eps=1e-12, hidden_dropout_prob=0.1, attention_probs_dropout=0.1):
        """
        Bert模型的配置
        :param vocab_size:词汇表的大小 词嵌入的维度
        :param hidden_size:隐藏层的大小
        :param num_hidden_layers:隐藏层的数量
        :param num_attention_heads:注意力头的个数
        :param intermediate_size:中间层大小 表示BERT模型中每个Transformer块中全连接层的隐藏单元数量。
        :param hidden_act:激活函数
        :param max_pos_embedding:最大位置嵌入 BERT模型能处理的最大序列长度
        :param type_vocab_size:类型词汇表大小 表示BERT模型中标记句子类型的标记数量。通常为2，表示句子A和句子B
        :param init_range:参数初始化范围
        :param layer_norm_eps:层归一化的eps
        :param hidden_dropout_prob:隐藏层dropout概率
        :param attention_probs_dropout:注意力dropout概率
        """

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_pos_embedding
        self.type_vocab_size = type_vocab_size
        self.initializer_range = init_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        """
        初始化Bert嵌入层
        :param config: Bert的配置
        """
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        前向传播
        :param input_ids: 输入的token id
        :param token_type_ids: 输入的token类型 id
        :return: 嵌入层的输出
        """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # 词嵌入，位置嵌入，类型嵌入
        input_embed = self.word_embeddings(input_ids)
        pos_embed = self.position_embeddings(
            torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embed + pos_embed + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    """
    Bert自注意力层 对于Transformer的自注意力层
    """
    def __init__(self, config):
        """
        初始化Bert的注意力机制
        :param config: 配置
        """
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose(self, x):
        """
        转置输入张量以便于计算得分
        :param x: 输入张量 (batch_size,seq_len,hidden_size)
        :return: 转换后的张量 (batch_size,num_attention_heads,seq_len,attention_head_size)
        """
        # 拿出x的batch_size,seq_len维度
        x_shape = x.size()[:-1]
        new_x_shape = x_shape + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        前向传播
        :param hidden_states: 输入的隐藏状态
        :param attention_mask: 注意力掩码
        :return: 自注意力的输出 (batch_size,seq_length,all_head_size)
        """
        # 线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 转置
        query_layer = self.transpose(mixed_query_layer)
        key_layer = self.transpose(mixed_key_layer)
        value_layer = self.transpose(mixed_value_layer)

        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask  # 屏蔽无效的注意力分数

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # (batch_size,num_attention_heads,seq_len,attention_head_size)
        context_layer = torch.matmul(attention_probs,value_layer)
        # (batch_size,seq_len,num_attention_heads,attention_head_size)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        # (batch_size,seq_len)
        context_layer_shape = context_layer.size()[:-2]
        # (batch_size,seq_len,self.all_head_size)
        new_context_layer_shape = context_layer_shape + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    """
    Bert自注意力输出层 将注意力输出还需要通过一个输出线性层
    """
    def __init__(self,config):
        """
        初始化
        :param config: 配置
        """
        super(BertSelfOutput,self).__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states,input_tensor):
        """
        前向传播
        :param hidden_states: 自注意力的输出
        :param input_tensor: 输入帐了
        :return: 自注意力输出层的输出
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input_tensor) # 残差连接+层归一化
        return hidden_states

class BertAttention(nn.Module):
    """
    Bert注意力层
    """
    def __init__(self,config):
        """
        初始化
        :param config: 配置
        """
        super(BertAttention,self).__init__()
        self.self_Attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,input,attention_mask):
        """
        前向传播
        :param input: 输入张量
        :param attention_mask: 注意力掩码
        :return: 注意力层的输出
        """
        self_output = self.self_Attention(input,attention_mask)
        attention_output = self.output(self_output,input)
        return attention_output


class BertIntermediate(nn.Module):
    """
    Bert中间层
    """
    def __init__(self,config):
        """
        初始化
        :param config: 配置
        """
        super(BertIntermediate,self).__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)

        if isinstance(config.hidden_act,str):
            """如果没有规定激活函数就用默认的gelu"""
            self.intermediate_act_fn = F.gelu
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self,hidden_states):
        """
        前向传播
        :param hidden_states: 输入的隐藏状态
        :return: 中间层输出
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    """Bert输出层"""
    def __init__(self,config):
        """
        初始化Bert输出层
        :param config: 配置
        """
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,hidden_states,input):
        """
        前向传播
        :param hidden_states: 中间层的输出
        :param input: 输入张量
        :return: 输出层的输出
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input) #残差连接+层归一化
        return hidden_states


class BertLayer(nn.Module):
    """Bert层"""
    def __init__(self,config):
        """
        初始化Bert层
        :param config: 配置
        """
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,hidden_states,attention_mask):
        """
        前向传播
        :param hidden_states: 输入的隐藏状态
        :param attention_mask: 注意力掩码
        :return: Bert层的输出
        """
        attention_output = self.attention(hidden_states,attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output,attention_output)
        return layer_output

class BertEncoder(nn.Module):
    """Bert编码器"""
    def __init__(self,config):
        """
        初始化Bert编码器
        :param config: 配置
        """
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,hidden_states,attention_mask):
        """
        前向传播
        :param hidden_states: 输入的隐藏状态
        :param attention_mask: 注意力掩码
        :return: 编码器的输出
        """
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states,attention_mask)
        return hidden_states

class BertCLS(nn.Module):
    """BertCLS"""
    def __init__(self, config):
        """
        初始化
        :param config: 配置
        """
        super(BertCLS, self).__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self,hidden_states):
        """
        前向传播
        :param hidden_states: 编码器的输出
        :return: 池化层得到输出
        """
        first_token = hidden_states[:,0]
        cls_output = self.dense(first_token)
        cls_output = self.activation(cls_output)
        return cls_output


class BertModel(nn.Module):
    """Bert模型"""
    def __init__(self,config):
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLS(config)

    def forward(self,input_ids,token_type_ids=None,attention_mask=None):
        """
        前向传播
        :param input_ids: 输入的id
        :param token_type_ids: 输入的token类型 id 例如A、B
        :param attention_mask: 注意力掩码
        :return: Bert模型的输出
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.ones_like(input_ids)

        # 生成扩展的注意力掩码
        """
        (1.0 - extended_attention_mask) * -10000.0：
        反转掩码（将非padding的位置设置为0，padding的位置设置为1），然后乘以一个非常大的负数（如-10000）
        这样在计算注意力得分时，padding位置的得分会非常低，几乎为0，从而不会对最终的输出产生影响。
        """
        # (batch_size,seq_len) -> (batch_size,1,1,seq_len)
        pad_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        pad_attention_mask = pad_attention_mask.to(dtype=torch.float32)
        pad_attention_mask = (1.0 - pad_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids,token_type_ids)
        encoder_output = self.encoder(embedding_output,pad_attention_mask)
        cls_output = self.cls(encoder_output)

        return encoder_output,cls_output


class BertPredictionHeadTransform(nn.Module):
    """Mask问题的transformer架构"""
    def __init__(self,config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.transform_loss_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_loss_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertMaskPredict(nn.Module):
    """针对Mask问题的预测"""
    def __init__(self,config):
        super(BertMaskPredict, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self,hidden_states):
        """
        前向传播
        :param hidden_states: 输入的隐藏状态
        :return: Mask模型的预测得分
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states



class BertPreTrainingScores(nn.Module):
    """计算预训练得分"""
    def __init__(self,config):
        super(BertPreTrainingScores, self).__init__()
        self.pred = BertMaskPredict(config)
        self.seq_relation = nn.Linear(config.hidden_size,2)

    def forward(self,seq_out,cls_out):
        """
        前向传播
        :param seq_out: 序列输出
        :param cls_out: cls输出
        :return: 预测得分和序列关系得分
        """
        prediction_scores = self.pred(seq_out)
        seq_relation_score = self.seq_relation(cls_out)
        return prediction_scores,seq_relation_score

class BertPreTraining(nn.Module):
    """Bert预训练 MLM MSP"""
    def __init__(self,config):
        super(BertPreTraining, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertPreTrainingScores(config)

    def forward(self,input_ids,token_type_ids=None,attention_mask=None, masked_lm_labels=None, next_sentence_labels=None):
        """
        :param input_ids:输入的token的id
        :param token_type_ids:输入的token类型的id
        :param attention_mask:注意力掩码
        :param masked_lm_labels:遮蔽模型的标签
        :param next_sentence_labels:下一句预测的标签
        :return:MLM和NSP任务的输出
        """
        outputs = self.bert(input_ids,token_type_ids,attention_mask)
        sequence_output,pool_output = outputs

        prediction_scores,seq_relation_score = self.cls(sequence_output,pool_output)

        loss = None
        if masked_lm_labels is not None and next_sentence_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            masked_loss = loss_fn(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fn(seq_relation_score.view(-1, 2), next_sentence_labels.view(-1))
            loss = masked_loss + next_sentence_loss

        return loss,prediction_scores,seq_relation_score


class BertSequenceClassify(nn.Module):
    """Bert微调：下游任务序列分类"""
    def __init__(self,config,num_labels):
        """
        初始化
        :param config: 配置
        :param num_labels: 类别数量
        """
        super(BertSequenceClassify,self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)

    def forward(self,input_ids,attention_mask,token_type_ids,labels=None):
        """
        前向传播
        :param input_ids: 输入词向量
        :param attention_mask: 注意力掩码
        :param token_type_ids: 输入词向量id
        :param labels:标签
        :return:损失和分类器的输出
        """
        outputs = self.bert(input_ids,attention_mask,token_type_ids)
        pool_output = outputs[1]
        pool_output = self.dropout(pool_output)
        logits = self.classifier(pool_output) # 分类器的输出

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits
        else:
            return logits


def train_seq_classify_model(model,dataloader,optimizer,epochs=3):
    """训练序列分类下游任务"""
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids,attention_mask,token_type_ids,labels = batch
            optimizer.zero_grad()
            loss,_ = model(input_ids,attention_mask,token_type_ids,labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs} with loss: {loss.item()}")


def BertPredict(model,dataloader):
    """Bert预测"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids,attention_mask,token_type_ids,_ = batch
            logits = model(input_ids,attention_mask,token_type_ids)
            preds = torch.argmax(logits,dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions


# 创建MLM任务的遮蔽输入
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    随机遮蔽输入文本中的一些单词，用于MLM任务
    :param inputs: 输入的token id
    :param tokenizer: BERT分词器
    :param mlm_probability: 遮蔽的概率
    :return: 遮蔽后的输入和标签
    """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 只有被遮蔽的tokens才计算loss

    # 80%的时间，替换为[MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10%的时间，替换为随机token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 剩下的10%的时间，保持原来的token
    return inputs, labels

# BERT预训练
def pre_train_bert(model, tokenizer, train_data, learning_rate=1e-4, batch_size=32, epochs=1):
    """
    预训练BERT模型
    :param model: BERT预训练模型实例
    :param tokenizer: BERT分词器
    :param train_data: 训练数据
    :param learning_reate: 学习率
    :param batch_size: 批次大小
    :param epochs: 训练轮数
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for step, (sentence_a, sentence_b, is_next) in enumerate(train_data):
            inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention_mask = inputs['attention_mask']

            input_ids, masked_lm_labels = mask_tokens(input_ids, tokenizer)
            next_sentence_labels = torch.LongTensor([int(is_next)])

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            masked_lm_labels=masked_lm_labels, next_sentence_labels=next_sentence_labels)
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}, Loss: {loss.item()}")

# # 示例用法
# config = BertConfig()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# pretraining_model = BertPreTraining(config)
#
# # 假设我们有一些训练数据
# pretraining_data = [
#     ("The man went to the store", "He bought a gallon of milk", True),
#     ("The man went to the store", "Penguins are flightless birds", False)
# ]
#
# pre_train_bert(pretraining_model, tokenizer, pretraining_data, learning_rate=1e-4, batch_size=2, epochs=1)
#
# # 示例数据和数据加载器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 加载预训练的BERT分词器
# texts = ["Example sentence 1", "Example sentence 2"]  # 示例文本数据
# labels = [0, 1]  # 对应的标签
#
# # 将文本数据转换为输入张量
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
# input_ids = inputs['input_ids']  # 输入的token id
# attention_mask = inputs['attention_mask']  # 注意力掩码
# token_type_ids = inputs['token_type_ids']  # token类型id
# labels = torch.tensor(labels)  # 转换标签为张量
#
# dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)  # 创建Tensor数据集
# dataloader = DataLoader(dataset, batch_size=2)  # 创建数据加载器
#
# # 定义优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # 使用AdamW优化器
#
# # 微调模型
# train_model(model, dataloader, optimizer, epochs=3)
#
# # 预测示例
# predictions = predict(model, dataloader)
# print(f"Predictions: {predictions}")