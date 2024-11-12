# coding=utf-8

import numpy as np
import torch
from torch._prims_common import Tensor
import torch.nn as nn


def get_len_mask(
    b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)

    for i in range(b):
        attn_mask[i, :, : feat_lens[i]] = 0

    return attn_mask.to(torch.bool)


def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(
        torch.bool
    )


def get_enc_dec_mask(
    b: int,
    max_feat_len: int,
    feat_lens: torch.Tensor,
    max_label_len: int,
    device: torch.device,
) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)

    for i in range(b):
        attn_mask[i, :, feat_lens[i] :] = 1

    return attn_mask.to(torch.bool)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.0):
        """
        d_model: 模型中输入和输出向量的维度大小
        d_k: Q 向量和 K 向量经过线性变换后的维度
        d_v: V 向量经过线性变换后的维度
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask: torch.Tensor, **kwargs):
        def tensor(t) -> torch.Tensor:
            return torch.Tensor(t)

        N = Q.size(0)  # N is the number of rows in Q or the number of queries
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # linear projection [N, seq_len, d_model] ->  [N, num_heads, seq_len, d_k or d_v]
        Q = tensor(self.W_Q(Q)).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = tensor(self.W_K(K)).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = tensor(self.W_V(V)).view(N, -1, num_heads, d_v).transpose(1, 2)

        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
            attn_mask = attn_mask.bool()

        scores: torch.Tensor = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
            attns = torch.softmax(scores, dim=-1)
            attns = self.dropout(attns)

        output = torch.matmul(attns, V)

        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output

class EncoderLayer(nn.Module):



class Transformer(nn.Module):
    pass


if __name__ == "__main__":
    # fmt: off
    '''
    Constants
    '''
    BATCH_SIZE = 16         # 一个训练轮次中模型接收的样本（句子）数量
    MAX_SEQ_LEN = 100       # the maximum length of input sequence（一个句子中单词的数量）
    FEAT_DIM = 80           # the dimension of input feature（一个单词的维度）
    hidden_dim = 512        # the dimension of hidden layer
    vocab_size = 26         # the size of vocabulary
    MAX_LABEL_LEN = 100     # the maximum length of output sequence

    '''
    Dummy data
    '''
    fbank_feature = torch.randn(BATCH_SIZE, MAX_SEQ_LEN, FEAT_DIM)  # input sequence
    feat_lens = torch.randint(1, MAX_SEQ_LEN, (BATCH_SIZE,))        # the length of each input sequence in the batch
    labels = torch.randint(0, 26, (BATCH_SIZE, MAX_LABEL_LEN))      # output sequence
    label_lens = torch.randint(1, 10, (BATCH_SIZE,))                # the length of each output sequence in the batch
    # fmt: on
