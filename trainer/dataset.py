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
        # 构建边矩阵：基于标签的共现关系，构建一个边矩阵，记录不同标签之间的关联强度
        # 边归一化：对边矩阵进行归一化处理，以便于模型处理
        # 特征提取：从Wikipedia中提取标签对应的特征（使用SBERT模型提取文本嵌入）
        
        label2id = {v: k for k, v in enumerate(mlb.classes_)}  # 创建一个从标签名称到索引的字典，这个索引用于访问边矩阵和特征矩阵。

        edges = torch.zeros((len(label2id), len(label2id))) # 用来记录标签之间的关联度。c*c
        
        for label in train_data["labels"]:  # 在训练数据中遍历 每个样本 的标签组合，为每对标签增加连接
            if len(label) >= 2: # 每个样本里面 对应的标签数量不同。 
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        src, tgt = label2id[label[i]], label2id[label[j]]
                        edges[src][tgt] += 1
                        edges[tgt][src] += 1

# 以下是对上述代码的理解
#######################################################################################################################################################
# edges矩阵将反映各个标签之间的共现频率，这种方法有效地将标签间的语义关系转化为模型可以理解的结构信息，从而在考虑到标签关联性的同时进行分类决策
# 假设我们有以下训练样本的标签数据：
# 样本1的标签: ["科技", "教育"]
# 样本2的标签: ["健康", "科技", "体育"]
# 样本3的标签: ["健康", "旅游"]
# 假定这些标签在label2id字典中的映射如下：

# "科技" -> 0
# "教育" -> 1
# "健康" -> 2
# "体育" -> 3
# "旅游" -> 4
# 边矩阵的构建过程
# 根据上面的标签数据，代码将如下执行：

# 样本1的标签是["科技", "教育"]：

# 这里有两个标签，所以它们之间会建立连接。
# src = label2id["科技"] = 0, tgt = label2id["教育"] = 1
# edges[0][1] 和 edges[1][0] 均增加 1。
# 样本2的标签是["健康", "科技", "体育"]：

# 这个样本有三个标签，我们需要为每对标签建立连接。
# src = label2id["健康"] = 2, tgt = label2id["科技"] = 0
# edges[2][0] 和 edges[0][2] 均增加 1。
# src = label2id["健康"] = 2, tgt = label2id["体育"] = 3
# edges[2][3] 和 edges[3][2] 均增加 1。
# src = label2id["科技"] = 0, tgt = label2id["体育"] = 3
# edges[0][3] 和 edges[3][0] 均增加 1。
# 样本3的标签是["健康", "旅游"]：

# 这里有两个标签，所以它们之间会建立连接。
# src = label2id["健康"] = 2, tgt = label2id["旅游"] = 4
# edges[2][4] 和 edges[4][2] 均增加 1
########################################################################################################################################################
        
        marginal_edges = torch.zeros((len(label2id))) # 创建一个记录每个标签在所有文档中出现次数的向量。
        
        for label in train_data["labels"]: # 计算每个样本里面 各个标签的出现次数
            for i in range(len(label)):
                marginal_edges[label2id[label[i]]] += 1
        
        for i in range(edges.size(0)): # 计算相关矩阵Aij ，使得矩阵的每个元素代表两个标签的相关性
            for j in range(edges.size(1)):
                if edges[i][j] != 0:
                    edges[i][j] = (edges[i][j] * len(train_data["labels"]))/(marginal_edges[i] * marginal_edges[j])  # len(train_data["labels"])是指有多少个样本

        
# 以上代码 是计算标签间的关联强度，并根据标签的共现频率和各自的出现频率来计算 标签间的相关性，使用的是一个修改版的点互信息（Point-wise Mutual Information, PMI）的概念，使得边矩阵中的每个元素准确反映两个标签的相关性。   
###########################################################################################################################################################
# 假设有以下的训练数据标签：

# 样本1的标签：["苹果", "香蕉"]
# 样本2的标签：["苹果", "橙子"]
# 样本3的标签：["香蕉", "橙子"]
# 样本4的标签：["苹果", "香蕉", "橙子"]
# 假设标签到ID的映射（label2id）如下：

# "苹果" -> 0
# "香蕉" -> 1
# "橙子" -> 2

# 初始化和计算marginal_edges

# 1. 初始化：
# marginal_edges = torch.zeros((len(label2id)))：这将创建一个长度为3的零向量（因为有三个标签），用于记录每个标签的出现次数。

# 2. 计算标签出现次数：

# 遍历每个样本的标签，统计每个标签的出现次数：
# 样本1：["苹果", "香蕉"] -> 苹果出现1次，香蕉出现1次
# 样本2：["苹果", "橙子"] -> 苹果再次出现，橙子出现1次
# 样本3：["香蕉", "橙子"] -> 香蕉再次出现，橙子再次出现
# 样本4：["苹果", "香蕉", "橙子"] -> 所有标签各再出现1次
# 经过统计，marginal_edges 更新为 [3, 3, 3]，即每个标签分别出现3次。

# 计算相关矩阵Aij
# 假设在之前的步骤中，edges已被计算）为：
# [[0, 2, 2],  # 苹果与香蕉共现2次，苹果与橙子共现2次
#  [2, 0, 1],  # 香蕉与苹果共现2次，香蕉与橙子共现1次
#  [2, 1, 0]]  # 橙子与苹果共现2次，橙子与香蕉共现1次

# 3. 计算相关矩阵Aij：
# 计算相关矩阵Aij，使得每个元素代表两个标签的相关性。相关性计算公式是 edges[i][j] = (edges[i][j] * len(train_data["labels"])) / (marginal_edges[i] * marginal_edges[j])。
# 在这个例子中，共有4个样本，所以 len(train_data["labels"]) 是4。
# 相关矩阵Aij计算例子：
# edges[0][1] = (2 * 4) / (3 * 3) = 8 / 9
# 类似地，所有其他的非零边也进行这样的计算

# 计算后的edges可能看起来类似于：
# [[0.0, 0.889, 0.889],
#  [0.889, 0.0, 0.444],
#  [0.889, 0.444, 0.0]]

# 这样的相关矩阵确保了标签间的关联强度是基于它们的共现概率和它们在数据集中的出现频率来衡量的。这对于后续的图卷积网络处理非常重要，因为它确保了信息的传递是基于实际的统计关联性进行的。

#################################################################################################################################################################

        
        edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id)))) # 对边矩阵加上单位矩阵（自连接），然后使用某种归一化方法处理边矩阵，确保信息正确传播
    
        # Get embeddings from wikipedia
        features = torch.zeros(len((label2id)), 768)  # 初始化一个特征矩阵，每个标签对应一个768维的向量（通常与使用的预训练模型如SBERT的输出维度相同）
        for label, id in tqdm(label2id.items()): # 从Wikipedia中获取每个标签的嵌入
            features[id] = get_embedding_from_wiki(self.sbert, label, n_sent=2)
            
        return edges, features  # 返回处理后的边矩阵和特征矩阵。边矩阵就是相关矩阵Aij

###################################################################################################################################################################
# 在之前的步骤中，归一化前的edges边矩阵已被计算（但未加自连接）如下：
# [[0.0, 0.889, 0.889],
#  [0.889, 0.0, 0.444],
#  [0.889, 0.444, 0.0]]


# 1. 添加自连接：
# edges + torch.diag(torch.ones(len(label2id)))，这里torch.diag(torch.ones(len(label2id)))生成一个大小为3x3的单位矩阵，然后与edges相加。
# 加上自连接后的边矩阵可能变为：
# [[1.0, 0.889, 0.889],
#  [0.889, 1.0, 0.444],
#  [0.889, 0.444, 1.0]]

# 2. 归一化处理：
# 使用某种归一化方法，如行归一化（每行元素之和为1）。这样做有助于在进行图卷积操作时，确保所有的输入特征向量都具有相同的尺度，使得训练过程更加稳定

# 获取标签的嵌入
# 3. 初始化特征矩阵
# features = torch.zeros(len(label2id), 768)：这行代码初始化一个零矩阵，其中每一行对应一个标签的768维向量。
# 4. 从Wikipedia获取标签嵌入
# 通过遍历label2id.items()，对每个标签使用get_embedding_from_wiki函数从Wikipedia抓取相关信息，并使用SBERT模型将该信息转化为一个768维的向量。
# n_sent=2可能表示为每个标签抓取两个句子的内容来生成向量

####################################################################################################################################################################

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
    
