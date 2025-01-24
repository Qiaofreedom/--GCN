import math
import torch
import torch.nn as nn

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):  # in_features（标签特征)、out_features（边矩阵，相关矩阵）以及bias（布尔值，决定是否添加偏置项）
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
    
    def forward(self, x, adj_matrix):  # x（输入特征矩阵）和adj_matrix（邻接矩阵也就是相关矩阵）
        sp = torch.matmul(x, self.weights)   # 计算输入特征矩阵和权重矩阵的矩阵乘法，得到中间特征表示。self.weights 是一个可学习的权重矩阵，它将每个节点的特征映射到一个新的空间，得到一个中间表示 sp
        output = torch.matmul(adj_matrix, sp)  # 将邻接矩阵与上一步计算得到的特征矩阵进行矩阵乘法，实现特征的邻域聚合。adj_matrix 是图的邻接矩阵，它表示节点间的连接关系（例如，若节点i和节点j有边相连，则 adj_matrix[i][j] = 1）论文里面有自己的计算方法。
        # 通过与 adj_matrix 的矩阵乘法，sp 中的每个节点特征被其邻居节点的特征加权平均。简单来说，邻居节点的特征会“传播”到目标节点上，这就是邻域聚合的关键部分。
        # 邻域融合的目的是利用图中节点之间的连接关系来丰富节点的特征表示，使得节点不仅仅依赖自己的特征，还能融入邻居节点的特征，从而捕捉图结构中的重要信息
        if self.bias is not None:
            return output + self.bias
        else:
            return output
