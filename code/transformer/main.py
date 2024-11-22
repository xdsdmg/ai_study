import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# 姑且把导包也放在这个地方吧


# S: 起始标记
# E: 结束标记
# P：意为padding，将当前序列补齐至最长序列长度的占位符
sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)  # 6
# 目标词典（包含特殊符）
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# 反向映射词典，idx ——> word
idx2word = {v: k for k, v in tgt_vocab.items()}
tgt_vocab_size = len(tgt_vocab)  # 9

src_len = 5  # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数
tgt_len = 6  # 输出序列dec_input/dec_output的最长序列长度


# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

print(' enc_inputs: \n', enc_inputs)  # enc_inputs: [2,5]
print(' dec_inputs: \n', dec_inputs)  # dec_inputs: [2,6]
print(' dec_outputs: \n', dec_outputs)  # dec_outputs: [2,6]


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        # 我们前面的enc_inputs.shape = [2,5],所以这个返回的是2
        return self.enc_inputs.shape[0]

        # 根据idx返回的是一组 enc_input, dec_input, dec_output

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


# 构建DataLoader
loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

# 用来表示一个词的向量长度
d_model = 512

# FFN的隐藏层神经元个数
d_ff = 2048

# 分头后的q、k、v词向量长度，依照原文我们都设为64
# 原文：queries and kes of dimention d_k,and values of dimension d_v .所以q和k的长度都用d_k来表示
d_k = d_v = 64

# Encoder Layer 和 Decoder Layer的个数
n_layers = 6

# 多头注意力中head的个数，原文：we employ h = 8 parallel attention layers, or heads
n_heads = 8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # dropout是原文的0.1，max_len原文没找到
        '''max_len是假设的一个句子最多包含5000个token'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_len * d_model 的矩阵，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维用来和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0)  # [5000,512] -> [1,5000,512]
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同


# 为enc_input和dec_input做一个mask，把占位符P的token（就是0） mask掉
# 返回一个[batch_size, len_q, len_k]大小的布尔张量，True是需要mask掉的位置
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k.data.eq(0)返回一个等大的布尔张量，seq_k元素等于0的位置为True,否则为False
    # 然后扩维以保证后续操作的兼容(广播)
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # pad_attn_mask: [batch_size,1,len_k]
    # 要为每一个q提供一份k，所以把第二维度扩展了q次
    # 另注意expand并非真正加倍了内存，只是重复了引用，对任意引用的修改都会修改原始值
    # 这里是因为我们不会修改这个mask所以用它来节省内存
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # return: [batch_size, len_q, len_k]
    # 返回的是batch_size个 len_q * len_k的矩阵，内容是True和False，
    # 第i行第j列表示的是query的第i个词对key的第j个词的注意力是否无意义，若无意义则为True，有意义的为False（即被padding的位置是True）


# 用于获取对后续位置的掩码，防止在预测过程中看到未来时刻的输入
# 原文：to prevent positions from attending to subsequent positions
def get_attn_subsequence_mask(seq):
    """seq: [batch_size, tgt_len]"""
    # batch_size个 tgt_len * tgt_len的mask矩阵
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu 是生成一个 upper triangular matrix 上三角矩阵，k是相对于主对角线的偏移量
    # k=1意为不包含主对角线（从主对角线向上偏移1开始）
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 因为只有0、1所以用byte节省内存
    return subsequence_mask  # return: [batch_size, tgt_len, tgt_len]
