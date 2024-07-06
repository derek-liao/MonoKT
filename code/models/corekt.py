import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = 1e-12
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class CoreKT(nn.Module):
    def __init__(self, num_skills, num_questions, embedding_size, num_blocks, dropout, kq_same, d_ff=256, 
            final_fc_dim=512, num_attn_heads=8, separate_qr=False, l2=1e-5):
        super().__init__()
        """
        Input:
            embedding_size: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.device = device
        self.num_skills = num_skills
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_questions
        self.l2 = l2
        self.separate_qr = separate_qr
        embed_l = embedding_size

        self.q_nn = nn.Sequential(
            nn.Linear(embedding_size,128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2)
        )
        # self.q_nn = nn.Linear(embedding_size, 2)
        self.s_nn = nn.Linear(embedding_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.fusion_mode = 'sum'
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.loss_fn = nn.CrossEntropyLoss()

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions + 1, 1)  # 题目难度
            self.q_embed_diff = nn.Embedding(self.num_skills + 1,
                                             embed_l, padding_idx=0)  # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)  # interaction emb, 同上

        # num_skills+1 ,embedding_size
        self.q_embed = nn.Embedding(self.num_skills+1, embed_l, padding_idx=0)
        if self.separate_qr:
            self.qa_embed = nn.Embedding(2 * self.num_skills + 1, embed_l)  # interaction emb
        else:  # false default
            self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(num_skills=num_skills, num_blocks=num_blocks, n_heads=num_attn_heads, dropout=dropout,
                                  embedding_size=embedding_size, d_feature=embedding_size / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same)

        self.out = nn.Sequential(
            nn.Linear(embedding_size + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 2)
        )
        # self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)
    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  embedding_size# c_ct
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, embedding_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
            
        return q_embed_data, qa_embed_data
    def multi_skills_embedding(self,c):
        concept_emb_sum = self.q_embed(c + 1).sum(-2)
        print(f'concept_emb_sum: {concept_emb_sum.size()}')
        # [batch_size, seq_len,1]
        concept_num = torch.where(c < 0, 0, 1).sum(-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def multi_diff_skills_embedding(self, c):
        concept_emb_sum = self.q_embed_diff(c + 1).sum(-2)
        # [batch_size, seq_len,1]
        concept_num = torch.where(c < 0, 0, 1).sum(-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def forward(self,feed_dict):
        cq = feed_dict['questions']
        rr = feed_dict['responses']
        cc = feed_dict['skills']
        cr = rr * (rr > -1).long()
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(cc, cr)

        pid_embed_data = None
        # if self.num_questions > 0:  # have problem id
        #     q_embed_diff_data = self.multi_diff_skills_embedding(cc)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        #     pid_embed_data = self.difficult_param(cq)  # uq 当前problem的难度
        #     q_embed_data = q_embed_data + pid_embed_data * \
        #                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(cc)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(cq)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                cr)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qr:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (
                                            qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2  # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,embedding_size
        # Pass to the decoder
        # output shape BS,seqlen,embedding_size or embedding_size//2
        d_output, s_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        q_logit = self.q_nn(q_embed_data.detach())[:,1:]
        s_logit = self.s_nn(s_output.detach())[:,:-1]
        logits = output[:,1:]
        # both q, k and v are the facts
        z_qks = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=True, s_fact=True)
        # q is the fact while k and v are the counter_factual
        z_q = self.fusion(logits, q_logit, s_logit, q_fact=True, k_fact=False, s_fact=False)
        # z_sk = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=True, s_fact=True)
        # z = self.fusion(logits, q_logit, s_logit, q_fact=False, k_fact=False, s_fact=False)
        logit_Core_AKT = z_qks - z_q  # TIE

        z_nde = self.fusion(logits.clone().detach(), q_logit.clone().detach(), s_logit.clone().detach(),
                            q_fact=True, k_fact=False, s_fact=False)
        # NDE = z_q - z
        true = rr[:, 1:].float()
        sm = true > -1
        sm_ = sm.unsqueeze(-1)
        z_nde_pred = torch.masked_select(z_nde, sm_).view(-1, 2)
        q_pred = torch.masked_select(q_logit, sm_).view(-1, 2)
        s_pred = torch.masked_select(s_logit, sm_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_qks, sm_).view(-1, 2)
        Core_AKT_pred = torch.masked_select(logit_Core_AKT, sm_).view(-1, 2)


        q = torch.masked_select(cq[:, 1:], sm)
        t = torch.masked_select(cr[:, 1:], sm).long()

        loss_cls = self.loss_fn(q_pred, t) + \
                   self.loss_fn(z_qks_pred, t)
        p_te = self.softmax(z_qks_pred).clone().detach()
        loss_kl = - p_te * self.softmax(z_nde_pred).log()
        loss_kl = loss_kl.sum(1).mean()
        loss = loss_cls + loss_kl+c_reg_loss

        out_dict = {
            "pred": self.softmax(Core_AKT_pred)[:, 1],
            "true": rr[:, 1:].float(),
            "loss": loss,
        }

        return out_dict

    def loss(self, feed_dict, out_dict):
        
        loss = out_dict['loss']
        return loss, 0, 0

    def fusion(self, z_k, z_q, z_s, q_fact=False, k_fact=False, s_fact=False):

        global z
        z_k, z_q, z_s = self.transform(z_k, z_q, z_s, q_fact, k_fact, s_fact)

        if self.fusion_mode == 'rubi':
            z = z_k * torch.sigmoid(z_q)

        elif self.fusion_mode == 'hm':
            z = z_k * z_s * z_q
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = z_k + z_q + z_s
            z = torch.log(torch.sigmoid(z) + eps)
        return z

    def transform(self, z_k, z_q, z_s, q_fact=False, k_fact=False, s_fact=False):

        if not k_fact:
            z_k = self.constant * torch.ones_like(z_k).to(self.device)

        if not q_fact:
            z_q = self.constant * torch.ones_like(z_q).to(self.device)

        if not s_fact:
            z_s = self.constant * torch.ones_like(z_s).to(self.device)

        if self.fusion_mode == 'hm':
            z_k = torch.sigmoid(z_k)
            z_q = torch.sigmoid(z_q)
            z_s = torch.sigmoid(z_s)

        return z_k, z_q, z_s





class Architecture(nn.Module):
    def __init__(self, num_skills, num_blocks, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            embedding_size : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = embedding_size
        """
        self.embedding_size = embedding_size

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(num_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(num_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)  # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data)  # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True,
                          pdiff=pid_embed_data)  # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x ,y


class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True,
                pdiff=pdiff)  # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.embedding_size = embedding_size

        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if kq_same is False:
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
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            # constant_(self.attnlinear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)

            # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * embedding_size

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        pdiff = None
        scores = attention(q, k, v, self.d_k,
                            mask, self.dropout, zero_pad, gammas, pdiff)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.embedding_size)

        output = self.out_proj(concat)

        return output

    def pad_zero(self, scores, bs, dim, zero_pad):
        if zero_pad:
            # # need: torch.Size([64, 1, 200]), scores: torch.Size([64, 200, 200]), v: torch.Size([64, 200, 32])
            pad_zero = torch.zeros(bs, 1, dim).to(device)
            scores = torch.cat([pad_zero, scores[:, 0:-1, :]], dim=1)  # 所有v后置一位
        return scores


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)  # 结果和上一步一样
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
        # print(f"distotal_scores: {disttotal_scores}")
        position_effect = torch.abs(
            x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)  # score <0 时，设置为0
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff == None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() *
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)