import os

import numpy as np
import torch
from torch import nn


from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from torch import Tensor
from typing import Optional
from enum import IntEnum



import math
import torch.nn.functional as fn
from torch.nn.functional import one_hot

class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            self.default_weight_init(linear.weight)
            self.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    # trunk model init
    def default_weight_init(self, tensor):
        torch.nn.init.xavier_uniform(tensor)
        # torch.nn.init.kaiming_normal_(tensor)


    def default_bias_init(self, tensor):
        torch.nn.init.constant_(tensor, 0)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Mamba4KT(Module):
    def __init__(self, num_skills, num_questions, embedding_size, num_attn_heads, num_blocks, d_state, d_conv, expand, dropout=0.1):
        super().__init__()

        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.num_blocks = num_blocks
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout= dropout
        self.mamba_states = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                bimamba_type='none',
                dropout=self.dropout,
                num_blocks=self.num_blocks,
            ) for _ in range(self.num_blocks)
        ])

        if self.num_questions > 0:
            self.question_difficult = nn.Embedding(self.num_questions + 1, self.embedding_size)
            self.concept_diff = nn.Embedding(self.num_skills + 1, self.embedding_size)
        self.concept_encoder = nn.Embedding(self.num_skills, self.embedding_size)
        self.answer_encoder = nn.Embedding(2, self.embedding_size)
        # self.state_encoder = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
        self.true_encoder = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
        self.false_encoder = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)
        self._mlp_trans = StackedDense(
            self.embedding_size,
            [self.hidden_size] * 2,
            ([torch.nn.Tanh] * (1)) + [None]
        )
        self.dual_attention = DualAttention(num_attn_heads=num_attn_heads, d_model=self.hidden_size)
        self.dropout_layer = Dropout(self.dropout)
        self.out_layer = Linear(self.hidden_size, self.num_skills)
        self.final_out = Linear(3 * self.hidden_size, self.hidden_size)
        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, feed_dict):
        '''
        c: [batch_size, seq_len]
        r: [batch_sze, seq_len]
        q: [batch_sze, seq_len]
        '''
        q = feed_dict['questions']
        c = feed_dict['skills']
        r = feed_dict['responses']
        cshft = c[:, 1:]
        rshft = r[:, 1:]
        masked_r = r * (r > -1).long()
        q_input = q[:, :-1]
        c_input = c[:, :-1]
        r_input = masked_r[:, :-1]
        concept_emb = self.concept_encoder(c_input)
        state = self.answer_encoder(r_input) + concept_emb
        if self.num_questions > 0: # have problem id
            concept_diff = self.concept_diff(c_input)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            question_difficult = self.question_difficult(q_input)  # uq 当前problem的难度
            concept_emb = concept_emb + question_difficult * concept_diff  # uq *d_ct + c_ct # question encoder
        # state = self.state_encoder(c + self.num_skills * r)
        state_true_emb = self.true_encoder(c_input * r_input + self.num_skills * r_input)
        state_false_emb = self.false_encoder(c_input * (1 - r_input))
        state = self._mlp_trans(state)
        state_true = self._mlp_trans(state_true_emb)
        state_false = self._mlp_trans(state_false_emb)
        y = state
        for i in range(self.num_blocks):
            y = self.mamba_states[i](y)
        # y, _ = self.lstm_layer(y)
        
        state_true, state_false = self.dual_attention(y, y, state_true, state_false)
        y = torch.cat((y, state_true, state_false), dim=-1)
        y = self.final_out(y)
        y = self.dropout_layer(y)
        y = self.out_layer(y)
        y = torch.sigmoid(y)
        y = (y * one_hot(cshft.long(), self.num_skills)).sum(-1)
        out_dict = {
            "pred": y,
            "true": rshft.float()
        }
        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()

class DualAttention(nn.Module):
    def __init__(self, num_attn_heads, d_model, dropout=0.1):
        super(DualAttention, self).__init__()
        assert d_model % num_attn_heads == 0
        self.num_attn_heads = num_attn_heads
        self.d_model = d_model
        self.n_feature = d_model // num_attn_heads
        self.dropout = nn.Dropout(p = dropout)
        # self.gammas = nn.Parameter(torch.zeros(num_attn_heads, 1, 1))
        # torch.nn.init.xavier_uniform_(self.gammas)
    
    def forward(self, q, k, v1, v2, mask=None):
        
        batch_size = q.size(0)
        device = q.device
        mask = create_mask(q, 0, device)
        q = q.view(batch_size, -1, self.num_attn_heads, self.n_feature)
        k = k.view(batch_size, -1, self.num_attn_heads, self.n_feature)
        v1 = v1.view(batch_size, -1, self.num_attn_heads, self.n_feature)
        v2 = v2.view(batch_size, -1, self.num_attn_heads, self.n_feature)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)
        scores, output_v1, attn_weight = attention(q, k, v1, mask, self.dropout)
        output_v2 = torch.matmul(attn_weight, v2)
        output_v1 = output_v1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_v2 = output_v2.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return output_v1, output_v2

 
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = fn.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)
    return scores, output, p_attn


def create_mask(input, mask, device):
    seqlen = input.size(1)
    nopeek_mask = np.triu(
    np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
    src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
    return src_mask




class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)

from torch.autograd import Variable

def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, bimamba_type, dropout, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.bimamba_type = bimamba_type
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = RMSNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):

        hidden_states = self.mamba(input_tensor)
        if self.num_blocks == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states
    
class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = RMSNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    