# -*- encoding: utf-8 -*-
import numpy as np
import random
import torch.nn as nn
import torch
import copy


class Attention_1(nn.Module):
    """
        First version of self-attention.
        Args: - local_embs: local embeddings, shape: (batch_size, L, embed_dim)
              - raw_global_emb: tagert embbeding, shape: (batch_size, embed_dim)
        Returns: - new_global: final embedding by self-attention, shape: (batch_size, embed_dim).
    """

    def __init__(self, embed_dim, with_ave=True, mul=False, common_type="two_layer"):
        """

        :param embed_dim:
        :param with_ave: if 'attention_noAve' == True: 之后加上平均值
        :param mul: 是否 global 与 local 相乘。
        common_type: "one_layer"
        """
        super().__init__()
        self.with_ave = with_ave
        self.mul = mul
        self.embed_dim = embed_dim
        if "two_layer" in common_type:
            self.embedding_common = nn.Sequential(
                nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
            )
        elif "one_layer" in common_type:
            self.embedding_common = nn.Sequential(
                nn.Linear(embed_dim, 1)
            )
        elif common_type == "meanpooling":
            self.embedding_common = "meanpooling"
        elif common_type == "maxpooling":
            self.embedding_common = "maxpooling"

        self.common_type = common_type

        self.softmax = nn.Softmax(dim=1)
        self.weights = 0  # attention 权重
        self.global_emb_weight_net = nn.Linear(1, 1, False)  # 存储 raw_global_emb 的权重
        self.change_raw_global_emb_weight(1)

    def get_raw_global_emb_weight(self):
        """
        得到 global_emb 的权重
        :return:
        """
        return self.global_emb_weight_net.weight.item()

    def change_raw_global_emb_weight(self, new_value: float):
        self.global_emb_weight_net.weight.data.fill_(new_value)

    def get_attention_weight(self):
        return torch.tensor(self.weights).clone().detach().cpu()

    def forward(self, local_embs: torch.Tensor, raw_global_emb=None):
        if self.embedding_common == "meanpooling":
            return torch.mean(local_embs, dim=1)
        elif self.embedding_common == "maxpooling":
            return torch.max(local_embs, dim=1).values
        if "+maxpooling" in self.common_type:
            raw_global_emb = torch.max(local_embs, dim=1).values

        if raw_global_emb is None:
            raw_global_emb = torch.mean(local_embs, dim=1)

        if self.mul:
            # compute the normalized weights, shape: (batch_size, L)
            g_emb = raw_global_emb.unsqueeze(1).repeat(1, local_embs.size(1), 1)
            local_embs = local_embs.mul(g_emb)  # (b, L, emb_size)

        weights = self.embedding_common(local_embs).squeeze(2)
        weights = self.softmax(weights)
        self.weights = weights

        # compute final text, shape: (batch_size, 1024)
        new_global = weights.unsqueeze(2) * local_embs
        if self.with_ave:
            new_global = new_global/2
            raw_global_emb = raw_global_emb/2
            new_global_weights = 1
            raw_global_weight = self.get_raw_global_emb_weight()
            self.weights = new_global_weights * weights + raw_global_weight * 1.0 / weights.shape[1]  # weights + meanpooling
            # compute final text
            new_global = new_global_weights * new_global + raw_global_weight * torch.unsqueeze(raw_global_emb, 1)


        new_global = new_global.sum(dim=1)

        return new_global
