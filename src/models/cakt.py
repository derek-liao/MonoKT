import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class CAKT(nn.Module):
    def __init__(self, num_skills, num_questions, embedding_size, num_blocks, dropout, kq_same, d_ff=256, final_fc_dim=512, num_attn_heads=8, separate_qa=False, l2=1e-5):
        super().__init__()
        """
        Input:
            embedding_size: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff: dimension for fully connected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.num_skills = num_skills
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_questions
        self.l2 = l2
        self.separate_qa = separate_qa
        self.temp = 0.05
        self.alpha = 0.1
        self.beta = 0.5
        embed_l = embedding_size
        
        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions+1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)
        
        self.q_embed = nn.Embedding(self.num_skills+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.num_skills+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        # Make sure d_feature is an integer
        d_feature = embedding_size // num_attn_heads
        self.model = Architecture(num_skills=num_skills, num_blocks=num_blocks, n_heads=num_attn_heads, dropout=dropout,
                                embedding_size=embedding_size, d_feature=d_feature, d_ff=d_ff, kq_same=self.kq_same)

        self.out = nn.Sequential(
            nn.Linear(embedding_size + embed_l, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        
        self.out_constra = nn.Sequential(
            nn.Linear(embedding_size*3, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        
        self.out_constra_bml = nn.Sequential(
            nn.Linear(embedding_size, final_fc_dim), 
            nn.ReLU(), 
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, embedding_size), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        self.loss_fn = nn.BCELoss(reduction="none")
        self.pearson = PearsonCorrelation()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, feed_dict):
        # Extract data from feed_dict
        q_data = feed_dict["skills"]
        r = feed_dict["responses"]
        pid_data = feed_dict["questions"] if self.num_questions > 0 else None
        attention_mask = feed_dict["attention_mask"]
        
        target = r * (r > -1).long()
        
        # Base embeddings
        q_embed_data = self.q_embed(q_data)  # BS, seqlen, embedding_size # c_ct
        if self.separate_qa:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data

        # Add question difficulty if available
        if self.num_questions > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # uq
            q_embed_data_diff = q_embed_data + pid_embed_data * q_embed_diff_data  # uq *d_ct + c_ct

            qa_data_flag = target
            qa_embed_data_diff = self.qa_embed(qa_data_flag) + pid_embed_data * q_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)

            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            q_embed_diff_data = self.q_embed_diff(q_data)
            q_embed_data_diff = q_embed_data + q_embed_diff_data
            qa_data_flag = target
            qa_embed_data_diff = self.qa_embed(qa_data_flag) + q_embed_data
            c_reg_loss = 0.

        # Get model outputs
        x_1 = q_embed_data_diff
        x_2 = q_embed_diff_data
        y_1 = qa_embed_data
        y_2 = q_embed_data
        z_1 = pid_embed_data * q_embed_diff_data + self.qa_embed(qa_data_flag) if self.num_questions > 0 else x_1
        z_2 = pid_embed_data * q_embed_diff_data if self.num_questions > 0 else x_2

        d_output, _ = self.model(x_1, x_2, y_1, y_2, z_1, z_2, n_pid=self.num_questions > 0)

        # Get final predictions
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        pred = torch.sigmoid(output)

        # Add contrastive learning components
        seqlength = d_output.size(1)
        bs = d_output.size(0)
        d_model_size = d_output.size(2)
        pad_zero = torch.zeros(bs, 1, d_model_size).to(d_output.device)
        d_output1 = torch.cat([pad_zero, d_output[:, :seqlength-1, :]], dim=1)
        d_output2 = torch.abs(d_output-d_output1)
        concat_contra = torch.cat([d_output, d_output1, d_output2], dim=-1)
        output_contra = self.out_constra(concat_contra).squeeze(-1)
        pred_contra = torch.sigmoid(output_contra)

        # BML components
        Q_data = torch.arange(self.num_skills).to(d_output.device)
        out_Q = self.q_embed(Q_data)
        Q = self.out_constra_bml(out_Q)
        flag1 = torch.tensor(1).to(d_output.device)
        flag0 = torch.tensor(0).to(d_output.device)
        q_pos = Q + self.qa_embed(flag1)
        q_neg = Q + self.qa_embed(flag0)

        # Calculate Pearson correlation
        d_output = d_output[:, :seqlength-1, :]
        d_output1 = d_output1[:, 1:seqlength, :]
        loss_pearson = self.pearson(d_output1, d_output)

        # Prepare output dictionary
        out_dict = {
            "pred": pred[:, 1:],
            "pred_contra": pred_contra[:, 1:],
            "true": r[:, 1:].float(),
            "c_reg_loss": c_reg_loss,
            "q_pos": q_pos,
            "q_neg": q_neg,
            "Q": Q,
            "loss_pearson": loss_pearson
        }

        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        pred_contra = out_dict["pred_contra"].flatten()
        true = out_dict["true"].flatten()
        c_reg_loss = out_dict["c_reg_loss"]
        q_pos = out_dict["q_pos"]
        q_neg = out_dict["q_neg"]
        Q = out_dict["Q"]
        loss_pearson = out_dict["loss_pearson"]
        
        mask = true > -1
        pred = pred[mask]
        pred_contra = pred_contra[mask]
        true = true[mask]
        
        loss = self.loss_fn(pred, true).mean()
        loss_contra = self.loss_fn(pred_contra, torch.ones_like(true)).mean()
        
        # Calculate contrastive loss
        contrastive_loss = ContrastiveLossELI5(self.num_skills, self.temp)(q_pos, q_neg)
        
        # Calculate BML loss
        bml_loss = BMLLoss(self.alpha, self.beta)(q_pos, q_neg, Q)
        
        total_loss = loss + loss_contra + c_reg_loss + contrastive_loss + bml_loss - loss_pearson
        
        return total_loss, len(pred), true.sum().item()

class Architecture(nn.Module):
    def __init__(self, num_skills, num_blocks, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=d_feature,
                           d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=True, kq_same=kq_same)
            for _ in range(num_blocks)
        ])
        
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=d_feature,
                           d_ff=d_ff, dropout=dropout, n_heads=n_heads, forget=True, kq_same=kq_same)
            for _ in range(num_blocks)
        ])

    def forward(self, x_1, x_2, y_1, y_2, z_1, z_2, n_pid):
        if n_pid:
            for block in self.blocks_1:
                x = block(mask=0, query=x_1, key=x_1, values=x_2)
                y = block(mask=0, query=y_1, key=y_1, values=y_2)
            
            for block in self.blocks_2:
                x = block(mask=0, query=x, key=x, values=z_1)
                y = block(mask=0, query=y, key=y, values=z_2)
            return x, y
        else:
            for block in self.blocks_1:
                x = block(mask=0, query=x_1, key=x_1, values=x_2)
                y = block(mask=0, query=y_1, key=y_1, values=y_2)
            
            for block in self.blocks_2:
                x = block(mask=1, query=x, key=x, values=y)
            return x, y

class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, d_feature,
                 d_ff, n_heads, dropout, forget, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, forget, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        device = query.device
        
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        
        return query

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, forget, kq_same, bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.forget = forget

        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
            
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if not self.kq_same:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout, 
                         zero_pad, self.gammas) if self.forget else attention_noforget(
                             q, k, v, self.d_k, mask, self.dropout, zero_pad, self.gammas)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_size)
        output = self.out_proj(concat)

        return output

def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    total_effect = torch.clamp(torch.clamp((dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

def attention_noforget(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones(
                (2 * self.batch_size, )).scatter_(0, torch.tensor([i]), 0.0).to(emb_i.device)
            denominator = torch.sum(
                one_for_not_i *
                torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            loss_ij = -torch.log(numerator / denominator)
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

class BMLLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.5, verbose=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

    def forward(self, emb_i, emb_j, emb_q):
        """
        emb_i: positive examples
        emb_j: negative examples
        emb_q: original knowledge points
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        z_q = F.normalize(emb_q, dim=1)
        similarity_pos = F.cosine_similarity(z_i, z_q)
        similarity_neg = F.cosine_similarity(z_j, z_q)
        diag_pos = torch.diag(similarity_pos)
        diag_neg = torch.diag(similarity_neg)
        temp = diag_neg-diag_pos
        loss = torch.relu(temp+self.alpha)+torch.relu(-temp-self.beta)
        loss_bml = torch.mean(loss)
        return loss_bml

class PearsonCorrelation(nn.Module):
    def forward(self, tensor_1, tensor_2):
        x = tensor_1
        y = tensor_2
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost 