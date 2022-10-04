''' Define the sublayers in encoder/decoder layer '''
import torch, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, rela_num = 0, relationE=None):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.rela_num = rela_num

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        if self.rela_num != 0:
            self.w_rs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vvs = nn.Linear(d_model, n_head * d_v, bias=False)
            self.relationE = relationE
            self.atta_dropout = nn.Dropout(0.1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, link=None):
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        if self.rela_num == 0: # Decode
            # Pass through the pre-attention projection: b x lq x (n*dv)
            # Separate different heads: b x lq x n x dv
            q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

            # Transpose for attention dot product: b x n x lq x dv
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q, attn = self.attention(q, k, v, mask=mask)
        else: # encode
            attn = self.calcE(residual, self.w_qs.weight, self.w_ks.weight, self.w_rs.weight, link) / self.d_k**0.5
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.atta_dropout(F.softmax(attn, dim=-1))
            q = self.calcZ(attn, residual, self.w_vs.weight, link)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().reshape(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

    def calcE(self, X, Wq, Wk, Wr, link):
        """
        X (M*)n*d
        Wq Wk Wr d*e
        link (M*)n*n*r
        relationE r*d
        return (M*)n*n
        """
        Xq = torch.matmul(X, Wq).reshape(X.size(0), X.size(1), self.n_head, -1).transpose(1,2) # batch, n_head, length of input, dim of embd
        Xk = torch.matmul(X, Wk).reshape(X.size(0), X.size(1), self.n_head, -1).transpose(1,2)
        Rk = torch.matmul(torch.matmul(link, self.relationE), Wr).reshape(X.size(0), X.size(1), X.size(1), self.n_head, -1).transpose(2,3).transpose(1,2) # batch, n_head, length of input, length of input, dim of embd
        rep = [1 for _ in range(len(Xk.size())+1)]
        rep[-3] = Xk.size(-2)
        Xk = Xk.unsqueeze(-3).repeat(rep) + Rk
        Xq = Xq.unsqueeze(-1)
        UP = torch.matmul(Xk, Xq).squeeze()
        return UP # batch, n_head, length of input, length of input

    def calcZ(self, alpha, X, Wv, link):
        """
        alpha (M*)n*n
        X (M*)n*d
        Wv d*e
        link (M*)n*n*r
        relationE r*d
        return (M*)n*e
        """
        Xv = torch.matmul(X ,Wv).reshape(X.size(0), X.size(1), self.n_head, -1).transpose(1,2) # batch, n_head, length of input, dim
        Rk = torch.matmul(torch.matmul(link, self.relationE), self.w_vvs.weight).reshape(X.size(0), X.size(1), X.size(1), self.n_head, -1).transpose(2,3).transpose(1,2) # batch, n_head, length of input, length of input, dim
        rep = [1 for _ in range(len(Xv.size())+1)]
        rep[-3] = Xv.size(-2)
        Xv = Xv.unsqueeze(-3).repeat(rep) + Rk
        Z = alpha.unsqueeze(-1) * Xv
        Z = Z.sum(-2)
        return Z # batch, n_head, length of input, dim


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
