import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, CrossEntropyLoss, MSELoss, BCELoss
import torch.nn.functional as F

class ATDKT(Module):
    def __init__(self, num_skills, num_questions, seq_len, emb_size=64, hidden_size=64, 
                 num_layers=1, dropout=0.1, emb_type='qid', num_attn_heads=5,
                 l1=0.5, l2=0.5, l3=0.5, start=50):
        super().__init__()
        self.model_name = "atdkt"
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.emb_type = emb_type
        self.start = start
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        # Main embeddings for skill+response interaction
        self.interaction_emb = Embedding(self.num_skills * 2, self.emb_size)
        
        # Question embeddings if needed
        if self.num_questions > 0:
            self.question_emb = Embedding(self.num_questions, self.emb_size)
            
        # Concept embeddings
        self.concept_emb = Embedding(self.num_skills, self.emb_size)
        
        # LSTM layer
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, 
                             num_layers=num_layers, batch_first=True)
        
        # Dropout
        self.dropout_layer = Dropout(dropout)
        
        # Output layers
        self.out_layer = Linear(self.hidden_size, self.num_skills)
        
        # Additional prediction layers
        self.qclassifier = Linear(self.hidden_size, self.num_skills)
        self.hisclassifier = Linear(self.hidden_size, 1)
        
        # Loss functions
        self.bce_loss = BCELoss(reduction='mean')
        self.ce_loss = CrossEntropyLoss()
        self.mse_loss = MSELoss()

    def forward(self, feed_dict):
        # Extract inputs
        q = feed_dict['skills']  # question/skill sequence
        r = feed_dict['responses']  # response sequence
        qshft = q[:, 1:]  # shifted questions
        rshft = r[:, 1:]  # shifted responses
        
        # Create interaction input
        x = q[:, :-1] + self.num_skills * (r[:, :-1] > 0).long()
        
        # Get embeddings
        xemb = self.interaction_emb(x)
        
        # Add question embeddings if available
        if self.num_questions > 0 and 'questions' in feed_dict:
            qemb = self.question_emb(feed_dict['questions'][:, :-1])
            xemb = xemb + qemb
            
        # Add concept embeddings
        cemb = self.concept_emb(q[:, :-1])
        xemb = xemb + cemb
        
        # LSTM processing
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        
        # Main prediction
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        
        # Get prediction for each skill
        y = (y * F.one_hot(qshft.long(), self.num_skills)).sum(-1)
        
        # Additional predictions
        qc_preds = self.qclassifier(h)  # concept prediction
        his_preds = torch.sigmoid(self.hisclassifier(h))  # history prediction
        
        out_dict = {
            "pred": y,
            "true": rshft,
            "qc_preds": qc_preds,
            "his_preds": his_preds,
        }
        return out_dict

    def loss(self, feed_dict, out_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        qshft = q[:, 1:]
        rshft = r[:, 1:]
        
        # Calculate mask for valid entries
        mask = rshft > -1
        
        # Main prediction loss
        y = out_dict["pred"]
        loss1 = self.bce_loss(y[mask], rshft[mask].float())
        
        # Concept prediction loss
        if self.l2 > 0:
            qc_preds = out_dict["qc_preds"]
            loss2 = self.ce_loss(qc_preds[mask], q[:, 1:][mask])
        else:
            loss2 = 0
            
        # History prediction loss
        if self.l3 > 0 and 'historycorrs' in feed_dict:
            his_preds = out_dict["his_preds"]
            start = self.start
            history_targets = feed_dict['historycorrs'][:, start:]
            history_mask = mask[:, start:]
            loss3 = self.mse_loss(his_preds[:, start:][history_mask], 
                                history_targets[history_mask])
        else:
            loss3 = 0
            
        # Combine losses
        loss = self.l1 * loss1
        if self.l2 > 0:
            loss = loss + self.l2 * loss2
        if self.l3 > 0:
            loss = loss + self.l3 * loss3
            
        return loss, mask.sum().item(), rshft[mask].sum().item() 