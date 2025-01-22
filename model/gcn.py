import math
import torch
import torch.nn as nn

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):  # in_features（输入特征的维度）、out_features（输出特征的维度）以及bias（布尔值，决定是否添加偏置项）
        super(GCNLayer, self).__init__()  # 调用父类的构造函数，确保正确初始化模块的层次结构
        self.in_features = in_features  # 将输入和输出特征的维度保存为  类的成员变量。
        self.out_features = out_features
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features)) # 定义一个权重矩阵，将其注册为模块的参数。nn.Parameter确保这些权重会在模型训练过程中被优化器更新。
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None) # 根据bias参数决定是否添加偏置项。如果bias为True，则为输出特征每个维度添加一个偏置参数；如果为False，则注册一个值为None的偏置参数。
        self.init_weights()
    
    def init_weights(self): # 调用init_weights方法来 初始化权重和偏置项。
        stdv = 1./math.sqrt(self.weights.size(1))   # 计算权重初始化时使用的标准差，这里使用 权重矩阵每列的大小（即输出特征的维度）来计算。
        self.weights.data.uniform_(-stdv, stdv)   # 使用均匀分布初始化权重，范围在-stdv到stdv之间。
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)   # 如果偏置项不为None，则同样使用均匀分布初始化偏置项。
    
    def forward(self, x, adj_matrix):  # x（输入特征矩阵）和adj_matrix（邻接矩阵）
        sp = torch.matmul(x, self.weights)   # 计算输入特征矩阵和权重矩阵的矩阵乘法，得到中间特征表示。
        output = torch.matmul(adj_matrix, sp)  # 将邻接矩阵与上一步计算得到的特征矩阵进行矩阵乘法，实现特征的邻域聚合。
        if self.bias is not None:
            return output + self.bias
        else:
            return output
