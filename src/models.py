import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query.unsqueeze(1)).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key.unsqueeze(1)).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value.unsqueeze(1)).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.W_o(attn_output).squeeze(1), attn_weights

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BCNet(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=6):
        super(BCNet, self).__init__()
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        if h_out is not None:
            if h_out <= self.c:
                self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
                self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
            else:
                self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if self.h_out is None:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            return torch.einsum('bvk,bqk->bvqk', (v_, q_))
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            return torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            return logits.transpose(2, 3).transpose(1, 2)

class DrugResponseModel(nn.Module):
    def __init__(self, cell_dim, d1_dim, d2_dim, hidden_dim=256):
        super(DrugResponseModel, self).__init__()
        self.cell_mlp = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim), nn.GELU(), nn.Dropout(0.3),
        )
        self.d1_proj = nn.Linear(d1_dim, hidden_dim)
        self.d2_proj = nn.Linear(d2_dim, hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim=hidden_dim, num_heads=4)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.bilinear_pool = BCNet(
            v_dim=hidden_dim, q_dim=hidden_dim,
            h_dim=hidden_dim, h_out=hidden_dim,
            dropout=[0.3, 0.5], k=6
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )
        self.calibration = nn.Linear(1, 1)
        nn.init.constant_(self.calibration.weight, 1.0)
        nn.init.constant_(self.calibration.bias, 0.0)

    def forward(self, c_feat, d_bertha, d_unimol):
        c_emb = self.cell_mlp(c_feat)
        d1 = F.gelu(self.d1_proj(d_bertha))
        d2 = F.gelu(self.d2_proj(d_unimol))
        a1, _ = self.cross_attn(d1, d2, d2)
        a2, _ = self.cross_attn(d2, d1, d1)
        drug_fused = self.layernorm(d1 + d2 + a1 + a2)
        bi_logits = self.bilinear_pool(c_emb.unsqueeze(1), drug_fused.unsqueeze(1))
        bi_feat = bi_logits.reshape(bi_logits.size(0), -1)
        raw_pred = self.predictor(torch.cat((c_emb, drug_fused, bi_feat), dim=1))
        return self.calibration(raw_pred)