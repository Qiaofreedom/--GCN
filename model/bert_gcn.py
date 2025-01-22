import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel  # AutoModel 是来自transformers库的，用于加载预训练的BERT模型

from model.gcn import GCNLayer

class BertGCN(nn.Module):
    def __init__(self, edges, features, config, args): # edges和features是分别代表边和特征
        super(BertGCN, self).__init__()
        self.label_features = features  # self.label_features存储标签特征
        self.edges = edges  # Aij, 相关矩阵里面的值
        self.device = args.device
        self.dropout = nn.Dropout(config['dropout_prob'])
        
        self.bert = AutoModel.from_pretrained(args.pretrained)  # 加载一个预训练的BERT模型。模型的具体类型由args.pretrained指定，这通常是一个模型的名称或路径。
        self.gc1 = GCNLayer(features.size(1), self.bert.config.hidden_size)  # 输入特征维度为 features的第二维大小，输出特征维度为 BERT模型隐藏层的大小。
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_features.size(0))  # 输入维度为BERT输出的隐藏状态维度，输出维度为 标签的数量。
        
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]  # input_ids和attention_mask是BERT模型输入所需的。这里从BERT模型获取最后一层的隐藏状态，取第一个token（通常是[CLS]标记）作为整个输入序列的表示，然后应用Dropout。
        bert_output = self.dropout(bert_output)  # 文本+Bert算法

        label_embed = self.gc1(self.label_features, self.edges) # 通过图卷积网络层处理 标签特征和边，获取 标签的嵌入表示 Label embedding（d'* c），并应用ReLU激活函数。
        label_embed = F.relu(label_embed) # 标签 + GCN算法

        output = torch.zeros((bert_output.size(0), label_embed.size(0)), device=self.device)  # 初始化一个输出张量，大小为批次大小d' 乘以 标签数量c，准备存储 每个文本样本 对应 每个标签的预测结果。
        
        for i in range(bert_output.size(0)):
            for j in range(label_embed.size(0)):
                output[i, j] = self.classifier(bert_output[i] + label_embed[j])[j] # 这个双层循环计算每个样本的BERT输出和每个标签嵌入的和，通过分类器得到最终的预测结果。每个样本和每个标签的组合都被单独计算
        return output
