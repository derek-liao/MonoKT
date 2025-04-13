import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, LayerNorm, Dropout, BCELoss, ModuleList, Parameter
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.nn import Sequential, ReLU
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)
    
def get_layer_norm(d_model):
    return LayerNorm(d_model)

class MoHAttention(Module):
    def __init__(self, d_model, d_feature, n_heads, n_shared_heads, 
                 n_selected_heads, dropout, kq_same,
                 seq_len=200, routing_mode="dynamic"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.h_shared = n_shared_heads
        self.h_selected = n_selected_heads
        self.kq_same = kq_same
        self.routing_mode = routing_mode
        
        # Linear layers for Q, K, V
        self.q_linear = Linear(d_model, d_model)
        if not kq_same:
            self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        
        # Routing networks
        if routing_mode == "dynamic":
            self.wg = nn.Linear(d_model, n_heads - n_shared_heads, bias=False)  # Router for dynamic heads
            
        self.dropout = Dropout(dropout)
        self.out = Linear(d_model, d_model)
        
        # Track routing statistics for load balancing
        self.register_buffer('head_selections', torch.zeros(n_heads - n_shared_heads))
        self.register_buffer('head_routing_probs', torch.zeros(n_heads - n_shared_heads))
        
    def get_balance_loss(self):
        # Calculate load balance loss for dynamic heads
        f = self.head_selections / (self.head_selections.sum() + 1e-5)
        P = self.head_routing_probs / (self.head_routing_probs.sum() + 1e-5)
        balance_loss = (f * P).sum()
        return balance_loss
        
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        seq_len = q.size(1)
        
        # Linear projections
        q = self.q_linear(q)  # [bs, seq_len, d_model]
        if self.kq_same:
            k = q
        else:
            k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Reshape for attention computation
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)  # [bs, h, seq_len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [bs, h, seq_len, seq_len]
        
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # First position zero padding
        pad_zero = torch.zeros(bs, self.h, 1, scores.size(-1)).to(q.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        
        # Calculate routing scores for dynamic heads
        q_for_routing = q.permute(0, 2, 1, 3).reshape(bs * seq_len, self.h * self.d_k)  # [bs*seq_len, h*d_k]
        
        # Handle dynamic heads routing
        if self.routing_mode == "dynamic":
            # Use learned routing weights
            logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
            gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
        else:  # query_norm mode
            # Calculate L2 norms for dynamic heads
            q_for_routing = q.permute(0, 2, 1, 3).reshape(-1, self.h, self.d_k)
            logits = torch.stack([
                torch.norm(q_for_routing[:, i, :], p=2, dim=1) 
                for i in range(self.h_shared, self.h)
            ], dim=1)  # [bs*seq_len, n_dynamic_heads]
            
            # Normalize logits
            logits_std = logits.std(dim=1, keepdim=True)
            logits_norm = logits / (logits_std / 1)
            gates = F.softmax(logits_norm, dim=1)  # [bs*seq_len, n_dynamic_heads]
        
        # Select top-k heads
        _, indices = torch.topk(gates, k=self.h_selected, dim=1)
        dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
        
        self.dynamic_scores = gates * dynamic_mask
        
        # Update routing statistics
        self.head_routing_probs = gates.mean(dim=0)
        self.head_selections = dynamic_mask.sum(dim=0)
        
        # Handle shared heads routing
        # All shared heads have equal weight of 1.0
        self.shared_scores = torch.ones(bs, seq_len, self.h_shared).to(q.device)
        
        dynamic_scores_reshaped = self.dynamic_scores.view(bs, seq_len, -1)
        routing_mask = torch.zeros(bs, seq_len, self.h).to(q.device)
        routing_mask[:, :, :self.h_shared] = 1.0  # Shared heads always active
        routing_mask[:, :, self.h_shared:] = dynamic_scores_reshaped  # Add dynamic head weights
        
        # Reshape routing mask to match attention dimensions [bs, h, seq_len, 1]
        routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        attn = self.dropout(torch.softmax(scores, dim=-1))
        
        # Save attention maps for visualization
        self.attention_maps = attn.detach().clone()  # [bs, h, seq_len, seq_len]
        
        context = torch.matmul(attn, v)  # [bs, h, seq_len, d_k]
        
        # Apply routing mask
        context = context * routing_mask
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(context)

class RouterKT(Module):
    def __init__(self, num_skills, num_questions, seq_len,
                 embedding_size, num_blocks, dropout, d_ff=256, 
                 num_attn_heads=8,
                 num_shared_heads=2,
                 num_selected_heads=4,
                 kq_same=True, final_fc_dim=512, final_fc_dim2=256, 
                 l2=1e-5,
                 separate_qr=False, balance_loss_weight=0.01,
                 routing_mode="dynamic",**kwargs):
        super().__init__()
        self.model_name = "routerkt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.dropout = dropout
        self.l2 = l2  # Store l2 parameter properly
        self.separate_qr = separate_qr
        self.balance_loss_weight = balance_loss_weight
        self.num_attn_heads = num_attn_heads
        self.num_shared_heads = num_shared_heads
        self.num_selected_heads = num_selected_heads
        self.routing_mode = routing_mode
        
        # Question difficulty parameters if using question IDs
        if self.num_questions > 0:
            self.difficult_param = Embedding(self.num_questions + 1, 1, padding_idx=0)  # μ_q
            self.q_embed_diff = Embedding(self.num_skills + 1, embedding_size, padding_idx=0)  # d_c
            self.qa_embed_diff = Embedding(2 * self.num_skills + 1, embedding_size, padding_idx=0)  # f_(c,r)
        
        # Basic embeddings
        self.q_embed = Embedding(num_skills, embedding_size, padding_idx=0)  # c_c
        if self.separate_qr:
            self.qa_embed = Embedding(2 * num_skills, embedding_size, padding_idx=0)  # 移除+1，与AKT对齐
        else:
            self.qa_embed = Embedding(2 + 1, embedding_size, padding_idx=0)  # 改为2+1维度，与AKT对齐
            
        # Validate head configurations
        if num_attn_heads < num_shared_heads:
            raise ValueError(f"Total attention heads ({num_attn_heads}) must be greater than shared heads ({num_shared_heads})")
        if num_selected_heads > (num_attn_heads - num_shared_heads):
            raise ValueError(f"Selected heads ({num_selected_heads}) cannot exceed number of dynamic heads ({num_attn_heads - num_shared_heads})")
            
        # Main architecture with MoH attention
        self.model = RouterKTArchitecture(
            n_question=num_skills,
            n_blocks=num_blocks,
            n_heads=num_attn_heads,  # Pass total number of heads
            n_shared_heads=num_shared_heads,  # Pass number of shared heads
            n_selected_heads=num_selected_heads,
            dropout=dropout,
            d_model=embedding_size,
            d_feature=embedding_size // num_attn_heads,  # Adjust feature dimension based on total heads
            d_ff=d_ff,
            kq_same=kq_same,
            seq_len=seq_len,
            routing_mode=routing_mode
        )
        
        # Output layers
        self.out = nn.Sequential(
            Linear(embedding_size * 2, final_fc_dim),
            nn.ReLU(),
            Dropout(dropout),
            Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            Dropout(dropout),
            Linear(final_fc_dim2, 1)
        )

        # self.out = Linear(embedding_size * 2, 1)

        
        self.loss_fn = BCELoss(reduction="mean")
        self.reset()
        
    def reset(self):
        """Reset parameters initialization."""
        for p in self.parameters():
            if p.size(0) == self.num_questions + 1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)
            elif p.dim() > 1:  # 添加对其他参数的初始化
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feed_dict):
        # Get current and previous timestep information
        q = feed_dict["skills"]
        r = feed_dict["responses"]
        attention_mask = feed_dict["attention_mask"]
        masked_r = r * (r > -1).long()
        pid_data = feed_dict["questions"] if "questions" in feed_dict else None
        diff = feed_dict["sdiff"] if "sdiff" in feed_dict else None
        if diff is not None:
            diff = (diff*(feed_dict["responses"]>-1).int()).long()
        
        # Get base embeddings for previous timesteps
        q_embed_data = self.q_embed(q)  # c_c
        if self.separate_qr:
            qa_data = q + self.num_skills * masked_r
            qa_embed_data = self.qa_embed(qa_data)  # 直接使用qa_embed
        else:
            qa_embed_data = q_embed_data + self.qa_embed(masked_r)  # c_c + g_r
            
        # Add question difficulty if using question IDs
        c_reg_loss = 0  # Initialize regularization loss
        if self.num_questions > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(q)  # d_c
            pid_embed_data = self.difficult_param(pid_data)  # μ_q
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data  # c_c + μ_q * d_c
            
            qa_embed_diff_data = self.qa_embed_diff(q + self.num_skills * masked_r)  # f_(c,r)
            qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
            
            # Calculate regularization loss
            c_reg_loss = torch.mean(pid_embed_data ** 2.0) * self.l2  # Use l2 parameter
            
        # Pass through transformer with MoH attention
        d_output, attn = self.model(q_embed_data, qa_embed_data, diff, r)
        
        # Final prediction
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = torch.sigmoid(self.out(concat_q)).squeeze(-1)
        
        out_dict = {
            "pred": output[:, 1:],
            "true": feed_dict["responses"][:, 1:].float(),
            "c_reg_loss": c_reg_loss
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        
        # Main prediction loss
        pred_loss = self.loss_fn(pred[mask], true[mask])
        
        # Add regularization loss
        c_reg_loss = out_dict["c_reg_loss"]
        
        # Add load balance loss from MoH layers
        balance_loss = self.model.get_balance_loss()
        
        # Total loss = prediction loss + regularization loss + balance loss
        total_loss = pred_loss + c_reg_loss + self.balance_loss_weight * balance_loss 
        
        return total_loss, len(pred[mask]), true[mask].sum().item()

class RouterKTArchitecture(Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, n_shared_heads, n_selected_heads, dropout,
                 kq_same, seq_len, routing_mode="dynamic"):
        super().__init__()
        self.d_model = d_model
        
        # Transformer blocks with MoH attention for knowledge encoder
        self.blocks_1 = ModuleList(
            [
                RouterTransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    n_shared_heads=n_shared_heads,
                    n_selected_heads=n_selected_heads,
                    kq_same=kq_same,
                    seq_len=seq_len,
                    routing_mode=routing_mode
                )
                for _ in range(n_blocks)
            ]
        )
        
        # Transformer blocks with MoH attention for question encoder
        self.blocks_2 = ModuleList(
            [
                RouterTransformerLayer(
                    d_model=d_model,
                    d_feature=d_feature,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    n_shared_heads=n_shared_heads,
                    n_selected_heads=n_selected_heads,
                    kq_same=kq_same,
                    seq_len=seq_len,
                    routing_mode=routing_mode
                )
                for _ in range(n_blocks * 2)
            ]
        )
        
    def forward(self, q_embed_data, qa_embed_data, diff=None, r=None):
        # Initialize variables
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)
        
        y = qa_embed_data
        x = q_embed_data
        
        # Knowledge encoder
        for block in self.blocks_1:
            y, _ = block(mask=1, query=y, key=y, values=y, diff=diff, response=r)
            
        # Question encoder
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                # x can see both current and past information
                x, _ = block(mask=1, query=x, key=x, values=x, diff=diff, response=r, apply_pos=False)
                flag_first = False
            else:# dont peek current response
                # knoweldge retriever
                # h can see past only
                x, attn = block(mask=0, query=x, key=x, values=y, diff=diff, response=r, apply_pos=True)
                flag_first = True
                
        return x, attn
    
    def get_balance_loss(self):
        balance_loss = 0
        for block in self.blocks_1:
            balance_loss += block.attn.get_balance_loss()
        for block in self.blocks_2:
            balance_loss += block.attn.get_balance_loss()
        return balance_loss

class RouterTransformerLayer(Module):
    def __init__(self, d_model, d_feature, d_ff, dropout, n_heads, 
                 n_shared_heads, n_selected_heads, kq_same, seq_len=200, routing_mode="dynamic", use_gradient_accumulation=False):
        super().__init__()
        
        # Pass parameters to MoHAttention
        self.attn = MoHAttention(
            d_model=d_model,
            d_feature=d_feature,
            n_heads=n_heads,
            n_shared_heads=n_shared_heads,
            n_selected_heads=n_selected_heads,
            dropout=dropout,
            kq_same=kq_same,
            seq_len=seq_len,
            routing_mode=routing_mode
        )
        
        # Layer norm and dropout
        self.layer_norm1 = get_layer_norm(d_model)
        self.dropout1 = Dropout(dropout)
        
        # Feed forward network
        self.ffn = transformer_FFN(d_model, dropout)
        self.layer_norm2 = get_layer_norm(d_model)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, mask, query, key, values, diff=None, response=None, apply_pos=True):
        # Create proper attention mask based on the mask parameter
        seqlen = query.size(1)
        if mask == 0:  # can only see past values
            # Create mask that only allows attention to past values
            nopeek_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=0).bool()
        else:  # mask == 1, can see current and past values
            # Create mask that allows attention to current and past values
            nopeek_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool()
            
        src_mask = (~nopeek_mask).to(query.device)
        
        # Apply MoH attention with proper masking
        attn_output = self.attn(query, key, values, src_mask)
        x = self.layer_norm1(query + self.dropout1(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))
        
        return x, self.attn.get_balance_loss()