import json
import math
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import *

class Dataset(object):
    
    def __init__(self,
                 train_data_path,
                 val_data_path,
                 test_data_path,
                 tokenizer,
                 batch_size,
                 max_length,
                 sbert): # Sentence-BERT模型（sbert）
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert
        
        self.train_loader, self.val_loader, self.test_loader, self.edges, self.label_features = self.load_dataset(train_data_path, val_data_path, test_data_path)
    
    def load_dataset(self, train_data_path, val_data_path, test_data_path):
        train = json.load(open(train_data_path))
        val = json.load(open(val_data_path))
        test = json.load(open(test_data_path))
        
        train_sents = [clean_string(text) for text in train['content']]  # 使用clean_string函数清洗数据中的文本
        val_sents = [clean_string(text) for text in val['content']]
        test_sents = [clean_string(text) for text in test['content']]

        
        mlb = MultiLabelBinarizer() # 使用MultiLabelBinarizer来转换标签数据为二进制形式，便于处理多标签分类问题
        train_labels = mlb.fit_transform(train['labels'])  # 将训练数据集中的标签转换为二进制形式，并训练mlb。
        print("Numbers of labels: ", len(mlb.classes_)) # 打印出标签的总数
        val_labels = mlb.transform(val['labels']) 
        test_labels = mlb.transform(test['labels'])

        edges, label_features = self.create_edges_and_features(train, mlb) # 基于 训练数据和标签 来创建 边和特征
        
        train_loader = self.encode_data(train_sents, train_labels, shuffle=True)  # 将训练数据编码并创建数据加载器，数据将被随机打乱
        val_loader = self.encode_data(val_sents, val_labels, shuffle=False)
        test_loader = self.encode_data(test_sents, test_labels, shuffle=False)
        
        return train_loader, val_loader, test_loader, edges, label_features
    
    def create_edges_and_features(self, train_data, mlb): # 创建一个从标签到ID的映射
        # 构建边：基于标签的共现关系，构建一个边矩阵，记录不同标签之间的关联强度
        # 边归一化：对边矩阵进行归一化处理，以便于模型处理
        # 特征提取：从Wikipedia中提取标签对应的特征（使用SBERT模型提取文本嵌入）
        
        label2id = {v: k for k, v in enumerate(mlb.classes_)}  # 创建一个从标签名称到索引的字典，这个索引用于访问边矩阵和特征矩阵。

        edges = torch.zeros((len(label2id), len(label2id))) # 用来记录标签之间的关联度
        
        for label in train_data["labels"]:  # 在训练数据中遍历每个样本的标签组合，为每对标签增加连接
            if len(label) >= 2:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        src, tgt = label2id[label[i]], label2id[label[j]]
                        edges[src][tgt] += 1
                        edges[tgt][src] += 1
        
        marginal_edges = torch.zeros((len(label2id))) # 创建一个记录每个标签在所有文档中出现次数的向量。
        
        for label in train_data["labels"]: # 计算每个标签的出现次数
            for i in range(len(label)):
                marginal_edges[label2id[label[i]]] += 1
        
        for i in range(edges.size(0)): # 对边矩阵进行归一化，使得矩阵的每个元素代表两个标签的相关性
            for j in range(edges.size(1)):
                if edges[i][j] != 0:
                    edges[i][j] = (edges[i][j] * len(train_data["labels"]))/(marginal_edges[i] * marginal_edges[j])
                    

        edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id)))) # 对边矩阵加上单位矩阵（自连接），然后使用某种归一化方法处理边矩阵，确保信息正确传播
    
        # Get embeddings from wikipedia
        features = torch.zeros(len((label2id)), 768)  # 初始化一个特征矩阵，每个标签对应一个768维的向量（通常与使用的预训练模型如SBERT的输出维度相同）
        for label, id in tqdm(label2id.items()): # 从Wikipedia中获取每个标签的嵌入
            features[id] = get_embedding_from_wiki(self.sbert, label, n_sent=2)
            
        return edges, features  # 返回处理后的边矩阵和特征矩阵
    
    def encode_data(self, train_sents, train_labels, shuffle=False):
        # 数据编码：使用tokenizer将文本数据转换为模型可以处理的格式（如，input ids和attention masks）
        # 数据封装：将编码后的文本数据和标签数据封装成Tensor，以便于在PyTorch中进行批处理。
        # DataLoader创建：创建DataLoader，用于在训练过程中批量加载数据，并根据参数选择是否随机打乱数据顺序
        
        X_train = self.tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        # self.tokenizer.batch_encode_plus：使用tokenizer批量处理传入的文本句子
        # padding=True：确保所有句子都填充到同一长度
        # truncation=True：确保所有句子长度不超过设定的最大长度self.max_length
        # max_length=self.max_length：设置句子的最大长度
        # return_tensors='pt'：返回PyTorch张量格式的数据
        
        y_train = torch.tensor(train_labels, dtype=torch.long) # 将标签数据train_labels转换为PyTorch张量，数据类型为长整型。
        
        train_tensor = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        # TensorDataset：利用PyTorch的TensorDataset创建一个数据集，这样可以方便地与DataLoader配合使用
        # X_train['input_ids']：包含了编码后的输入ID
        # X_train['attention_mask']：包含了注意力掩码，指示哪些部分是填充的，哪些部分是实际数据。
        
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        
        return train_loader
    
