import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class MLPExperts(nn.Module):
    def __init__(self, dim, mlp_layer=1, num_experts=16, output_dim=None, dpo=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dpo)
        self.num_experts = num_experts
        self.mlp_layer_num = mlp_layer
        self.output_dim = output_dim
        
        experts_mlp = []
        experts_out = []

        for e in range(num_experts):
            temp_lins = nn.ModuleList([
                nn.Linear(dim, dim)
                for _ in range(mlp_layer)
            ])
            experts_mlp.append(temp_lins)
            temp_out = nn.Linear(dim, output_dim)
            experts_out.append(temp_out)

        self.experts_mlp = nn.ModuleList(experts_mlp)
        self.experts_out = nn.ModuleList(experts_out)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        final_out = []
        for e in range(self.num_experts):
            cur_e_in = x[e]
            cur_e_mlp = self.experts_mlp[e]
            cur_e_out_layer = self.experts_out[e]
            
            for layer in cur_e_mlp:
                cur_e_in = F.relu(self.layer_norm(layer(cur_e_in)))
            
            cur_e_out = cur_e_out_layer(self.dropout(cur_e_in))
            final_out.append(cur_e_out)
        
        return torch.stack(final_out)

class AggregateMoE(nn.Module):
    def __init__(self, input_dim, num_experts=16, experts=None):
        super().__init__()
        self.num_experts = num_experts
        self.experts = experts
        self.input_dim = input_dim
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, inputs):
        gate_distribution = self.gate(inputs).softmax(dim=-1)
        all_expert_inputs = inputs.unsqueeze(0).repeat(self.num_experts, 1, 1, 1)
        all_expert_outputs = self.experts(all_expert_inputs)
        aggregate_outputs = torch.einsum("ebsd,bse->bsd", all_expert_outputs, gate_distribution)
        return aggregate_outputs

class QIKTMOE(Module):
    def __init__(self, num_skills, num_questions, seq_len, embedding_size=64, num_experts=16, dropout=0.1, **kwargs):
        super().__init__()
        self.num_skills = num_skills 
        self.num_questions = num_questions
        self.emb_size = embedding_size
        self.hidden_size = embedding_size
        self.num_experts = num_experts
        
        # Question interaction embedding
        self.interaction_emb = Embedding(self.num_questions * 2, self.emb_size)
        
        # LSTM layers
        self.que_lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.concept_lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        
        self.dropout_layer = Dropout(dropout)

        # MoE layers
        self.out_question_all_experts = MLPExperts(
            dim=self.hidden_size,
            mlp_layer=1,
            num_experts=num_experts,
            output_dim=num_questions,
            dpo=dropout
        )
        
        self.out_question_all = AggregateMoE(
            input_dim=self.hidden_size,
            num_experts=num_experts,
            experts=self.out_question_all_experts
        )

        self.out_concept_all_experts = MLPExperts(
            dim=self.hidden_size,
            mlp_layer=1,
            num_experts=num_experts,
            output_dim=num_skills,
            dpo=dropout
        )
        
        self.out_concept_all = AggregateMoE(
            input_dim=self.hidden_size,
            num_experts=num_experts,
            experts=self.out_concept_all_experts
        )

        self.loss_fn = nn.BCELoss(reduction="mean")

    def forward(self, feed_dict):
        q = feed_dict['skills']  # question sequence
        r = feed_dict['responses']  # response sequence
        
        # Process input
        masked_r = r * (r > -1).long()
        q_input = q[:, :-1]  # Remove last question
        r_input = masked_r[:, :-1]  # Remove last response
        q_shft = q[:, 1:]  # Remove first question
        r_shft = r[:, 1:]  # Remove first response

        # Get interaction embeddings
        x = q_input + self.num_questions * r_input
        xemb = self.interaction_emb(x)
        
        # Question path
        que_h = self.dropout_layer(self.que_lstm_layer(xemb)[0])
        q_out = self.out_question_all(que_h)
        y_q = torch.sigmoid(q_out)
        y_q = (y_q * F.one_hot(q_shft.long(), self.num_questions)).sum(-1)
        
        # Concept path  
        concept_h = self.dropout_layer(self.concept_lstm_layer(xemb)[0])
        c_out = self.out_concept_all(concept_h)
        y_c = torch.sigmoid(c_out)
        y_c = (y_c * F.one_hot(q_shft.long(), self.num_skills)).sum(-1)
        
        # Combine predictions
        y = (y_q + y_c) / 2
        
        out_dict = {
            "pred": y,
            "true": r_shft.float()
        }
        
        return out_dict

    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask])
        return loss, len(pred[mask]), true[mask].sum().item()
