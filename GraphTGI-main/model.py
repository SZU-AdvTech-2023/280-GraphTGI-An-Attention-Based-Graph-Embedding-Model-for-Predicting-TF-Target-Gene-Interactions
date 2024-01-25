import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
from mxnet.gluon.nn import activations
from mxnet.ndarray.gen_op import Activation
import numpy as np
from dgl.nn import GATConv
from dgl.nn import NNConv


# 对TF（转录因子）节点的输入特征进行投影
class TFEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(TFEmbedding, self).__init__()

        # 创建一个序列模块seq用于添加全连接层和dropout层
        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_TF = seq

    def forward(self, ndata):
        # 对TF节点的输入特征进行线性映射和dropout处理
        extra_repr = self.proj_TF(ndata['TF_features'])

        return extra_repr


# 对tg（靶基因）节点的输入特征进行投影
class tgEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(tgEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_tg = seq

    def forward(self, ndata):
        extra_repr = self.proj_tg(ndata['tg_features'])
        return extra_repr


def edge_func(efeat):
    return mx.nd.LeakyReLU(efeat, slope=0.5)


# 对图进行编码
class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        # 初始化图G
        self.G = G
        self.TF_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.tg_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()

        in_feats = embedding_size

        # 进行多层图注意力机制的计算 使用两层GATConv
        self.layers.add(
            GATConv(
                embedding_size,  # 每个节点的输入维度（嵌入大小）
                embedding_size,  # 每个节点的输出维度（嵌入大小）
                2,  # 注意力头的数量
                feat_drop=dropout,  # 节点特征的丢弃方法
                attn_drop=0.5,  # 注意力权重的丢弃率
                negative_slope=0.5,  # LeakyReLU激活函数的负斜率
                residual=True,  # 是否使用残差连接
                allow_zero_in_degree=True  # 是否允许节点度数为零
            )
        )
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop=dropout, attn_drop=0.5, negative_slope=0.5,
                                residual=True, allow_zero_in_degree=True))

        # 进行多层图注意力机制的计算 使用NNConv
        # self.layers.add(
        #     NNConv(
        #         in_feats=embedding_size,  # 输入特征的大小（每个节点的嵌入大小）
        #         out_feats=embedding_size,  # 输出特征的大小（每个节点的嵌入大小）
        #         edge_func=edge_func,  # 映射每个边特征的函数
        #         aggregator_type='mean',  # 聚合器类型，可以是 'sum', 'mean' 或 'max'
        #         residual=True,  # 是否使用残差连接
        #         bias=True  # 是否添加可学习的偏置
        #     )
        # )


        # 使用TF Embedding和tg Embedding 方法对 TF节点和目标基因节点进行特征投影
        self.TF_emb = TFEmbedding(embedding_size, dropout)
        self.tg_emb = tgEmbedding(embedding_size, dropout)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()

        # 将TF节点和目标基因节点的特征投影应用到图G上
        G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)
        G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)

        # 循环遍历将每一层的计算应用到图G上
        for layer in self.layers:
            layer(G, G.ndata['h'])

        return G.ndata['h']


# 对 TF 和靶基因节点的嵌入向量进行bilinear 解码操作
class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        # 使用sigmoid激活函数
        self.activation = nn.Activation('sigmoid')
        # 获取输入特征转化为更高级特征后的权值矩阵
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_TF, h_tg):
        # 对TF 节点的嵌入向量 h_TF 和目标基因节点的嵌入向量 h_tg 进行bilinear 解码操作
        results_mask = self.activation((nd.dot(h_TF, self.W.data()) * h_tg).sum(1))

        return results_mask


# 构建图神经网络模型
class GraphTGI(nn.Block):
    def __init__(self, encoder, decoder):
        super(GraphTGI, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, TF, tg):
        # 获取图G的节点特征
        h = self.encoder(G)

        # 获取TF节点和目标基因节点的特征向量
        h_TF = h[TF]
        h_tg = h[tg]

        return self.decoder(h_TF, h_tg), G.ndata['h']
