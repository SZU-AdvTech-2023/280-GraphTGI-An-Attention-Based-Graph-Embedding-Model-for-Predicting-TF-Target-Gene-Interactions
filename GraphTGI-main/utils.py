import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl


def load_data(directory):
    # 作为特征的化学相似性
    TFSM_1 = np.loadtxt('./data/chemical_similarity/TF_TF_chemical_similarity.txt')
    tgSM_1 = np.loadtxt('./data/chemical_similarity/tg_tg_chemical_similarity.txt')
    TF_tg_SM_1 = np.loadtxt('./data/chemical_similarity/TF_tg_chemical_similarity.txt')
    tg_TF_SM_1 = np.loadtxt('./data/chemical_similarity/tg_TF_chemical_similarity.txt')

    # 作为特征的序列相似性
    TFSM_2 = np.loadtxt('./data/seq_similarity/TF_TF_seq_similarity.txt')
    tgSM_2 = np.loadtxt('./data/seq_similarity/tg_tg_seq_similarity.txt')
    TF_tg_SM_2 = np.loadtxt('./data/seq_similarity/TF_tg_seq_similarity.txt')
    tg_TF_SM_2 = np.loadtxt('./data/seq_similarity/tg_TF_seq_similarity.txt')

    return TFSM_1, tgSM_1, TF_tg_SM_1, tg_TF_SM_1, TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2


def sample(directory, random_seed):
    # 读取包含 TF-tg 关联的 CSV 文件
    all_associations = pd.read_csv('./data/all_TF_tg_pairs.csv', names=['TF', 'tg', 'label'])

    # 获取所有正样本
    known_associations = all_associations.loc[all_associations['label'] == 1]

    # 随机获取相同数量的负样本
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    # 合并正负样本并重置索引
    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df.values


def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    # 加载化学相似性和序列相似性的数据矩阵
    TFSM, tgSM, TF_tg_SM, tg_TF_SM, TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2 = load_data(directory)
    # 从正样本和负样本中抽取相同数量的样本
    samples = sample(directory, random_seed)

    print('Building graph ...')
    # 创建一个多图对象 g1，初始时不包含节点和边
    g1 = dgl.DGLGraph(multigraph=True)
    # 添加节点，节点数量等于化学相似性和序列相似性矩阵的列数之和
    g1.add_nodes(TFSM.shape[1] + tgSM.shape[1])
    # 创建节点类型的ndarray，其中节点类型为TF的节点被标记为1，其余为0
    node_type = nd.zeros(g1.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:TFSM.shape[1]] = 1
    # 将节点类型数据拷贝到图的计算设备（CPU或GPU）上
    g = g1.to(ctx)
    g.ndata['type'] = node_type

    # 特征融合 将化学相似性和序列相似性的特征按列拼接
    print('Adding TF features ...')
    # 添加节点特征
    TF_data = nd.zeros(shape=(g.number_of_nodes(), TFSM.shape[1] + TFSM_2.shape[1]), dtype='float32', ctx=ctx)
    TF_data[:TFSM.shape[0], :TFSM.shape[1]] = nd.from_numpy(TFSM)
    TF_data[:TFSM.shape[0], TFSM.shape[1]:TFSM.shape[1] + TFSM_2.shape[1]] = nd.from_numpy(TFSM_2)
    TF_data[TFSM.shape[0]: TFSM.shape[0] + tgSM.shape[0], :tg_TF_SM.shape[1]] = nd.from_numpy(tg_TF_SM)
    TF_data[TFSM.shape[0]: TFSM.shape[0] + tgSM.shape[0],
    tg_TF_SM.shape[1]:tg_TF_SM.shape[1] + tg_TF_SM_2.shape[1]] = nd.from_numpy(tg_TF_SM_2)
    g.ndata['TF_features'] = TF_data

    print('Adding target gene features ...')
    # 添加靶基因特征
    tg_data = nd.zeros(shape=(g.number_of_nodes(), tgSM.shape[1] + tgSM_2.shape[1]), dtype='float32', ctx=ctx)
    tg_data[:TFSM.shape[0], :TF_tg_SM.shape[1]] = nd.from_numpy(TF_tg_SM)
    tg_data[:TFSM.shape[0], TF_tg_SM.shape[1]:TF_tg_SM.shape[1] + TF_tg_SM_2.shape[1]] = nd.from_numpy(TF_tg_SM_2)
    tg_data[TFSM.shape[0]: TFSM.shape[0] + tgSM.shape[0], :tgSM.shape[1]] = nd.from_numpy(tgSM)
    tg_data[TFSM.shape[0]: TFSM.shape[0] + tgSM.shape[0],
    tgSM.shape[1]:tgSM.shape[1] + tgSM_2.shape[1]] = nd.from_numpy(tgSM_2)
    g.ndata['tg_features'] = tg_data

    print('Adding edges ...')
    # 生成TF和tg的ID列表
    TF_ids = list(range(1, TFSM.shape[1] + 1))
    tg_ids = list(range(1, tgSM.shape[1] + 1))

    # 创建ID到索引的逆映射字典
    TF_ids_invmap = {id_: i for i, id_ in enumerate(TF_ids)}
    tg_ids_invmap = {id_: i for i, id_ in enumerate(tg_ids)}

    # 获取样本中的TF和tg节点索引
    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]
    sample_tg_vertices = [tg_ids_invmap[id_] + TFSM.shape[0] for id_ in samples[:, 1]] #

    # 根据从正样本和负样本中抽取的样本数据，将对应的TF和靶基因之间添加边。边的数据包括'inv'表示边的方向，以及'rating'表示正样本或负样本的标签
    g.add_edges(sample_TF_vertices, sample_tg_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})

    g.add_edges(sample_tg_vertices, sample_TF_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})

    # 将图设置为只读，以避免在训练过程中意外修改图的结构
    g.readonly()
    print('Successfully build graph !!')

    return g, TF_ids_invmap, tg_ids_invmap, TFSM
