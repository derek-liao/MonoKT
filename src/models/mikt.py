import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIKT(nn.Module):
    def __init__(self, num_skills, num_questions, seq_len, pro2skill, state_d, embedding_size, dropout):
        super(MIKT, self).__init__()

        self.num_skills = num_skills
        self.num_questions = num_questions

        self.pro_embed = nn.Parameter(torch.rand(num_questions, embedding_size))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(num_skills, embedding_size))
        nn.init.xavier_uniform_(self.skill_embed)

        self.var = nn.Parameter(torch.rand(num_questions, embedding_size))
        self.change = nn.Parameter(torch.rand(num_questions, 1))

        d = embedding_size
        state_d = state_d
        max_seq = seq_len

        self.pos_embed = nn.Parameter(torch.rand(max_seq, embedding_size))
        nn.init.xavier_uniform_(self.pos_embed)

        state_embed = state_d
        self.pro2skill = pro2skill
        self.skill_state = nn.Parameter(torch.rand(num_skills, state_embed))
        self.time_state = nn.Parameter(torch.rand(max_seq, state_embed))
        self.all_state = nn.Parameter(torch.rand(1, state_embed))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * state_embed, state_embed),
            nn.ReLU(),
            nn.Linear(state_embed, state_embed),
            nn.Sigmoid()
        )

        self.ans_embed = nn.Embedding(2, d)
        self.lstm = nn.LSTM(2 * d, d, batch_first=True)

        self.now_obtain = nn.Sequential(
            nn.Linear(d, state_embed),
            nn.Tanh(),
            nn.Linear(state_embed, state_embed),
            nn.Tanh()
        )

        self.pro_diff_embed = nn.Parameter(torch.rand(num_questions, d))
        self.pro_diff = nn.Embedding(num_questions, 1)

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)

        self.pro_guess = nn.Embedding(num_questions, 1)
        self.pro_divide = nn.Embedding(num_questions, 1)

        self.skill_state = nn.Parameter(torch.rand(num_skills, state_embed))

        self.pro_ability = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self.obtain1_linear = nn.Linear(d, d)
        self.obtain2_linear = nn.Linear(d, d)

        self.pro_diff_judge = nn.Linear(d, 1)

        self.all_obtain = nn.Linear(d, d)

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d)
        )

        self.do_attn = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

        self.predict_attn = nn.Linear(3 * d, d)

        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        self.loss_fn = nn.BCELoss()


    def forward(self, feed_dict):
        q = feed_dict['questions']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()
        last_problem = q[:, :-1]
        last_ans = masked_r[:, :-1]
        next_problem = q[:, 1:]
        next_ans = masked_r[:, 1:].long()
        device = last_problem.device

        seq = last_problem.shape[1]
        batch = last_problem.shape[0]

        pro2skill = self.pro2skill.to(device)

        pro_embed = self.pro_embed
        skill_embed = self.skill_embed

        skill_mean = torch.matmul(pro2skill, skill_embed) / (
                torch.sum(pro2skill, dim=-1, keepdims=True) + 1e-8)  # pro d

        pro_idx = torch.arange(self.num_questions).to(device)
        pro_diff_embed = F.embedding(pro_idx, self.pro_diff_embed)  # pro_max d

        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))  # pro_max 1

        q_pro = self.pro_linear(pro_embed)
        q_skill = self.skill_linear(self.skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, pro2skill == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, skill_embed)

        now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)

        pro_embed = self.dropout(now_embed)

        # pro_diff = torch.sigmoid(self.pro_diff_judge(pro_embed))  # pro_max 1
        contrast_loss = 0

        next_origin_q = F.embedding(next_problem, pro_embed)

        # pro_diff = torch.sigmoid(self.pro_diff(pro_embed)) # pro_max 1

        last_pro_rasch = F.embedding(last_problem, pro_embed)
        next_pro_rasch = F.embedding(next_problem, pro_embed)
        next_pro_diff = F.embedding(next_problem, pro_diff)
        next_pro_guess = torch.sigmoid(self.pro_guess(next_problem))
        next_pro_divide = self.pro_divide(next_problem)

        next_X = next_pro_rasch + self.ans_embed(next_ans)
        X = last_pro_rasch + self.ans_embed(last_ans)

        last_all_time = torch.ones((batch)).to(device).long()

        time_embed = self.time_state  # seq d
        all_gap_embed = F.embedding(last_all_time, time_embed)  # batch d

        res_p = []

        last_skill_time = torch.zeros((batch, self.num_skills)).to(device).long()  # batch skill

        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch skill d

        all_state = self.all_state.repeat(batch, 1)  # batch d
        batch_indices = torch.arange(batch).to(device)

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]  # batch
            now_pro2skill = F.embedding(now_pro, pro2skill).unsqueeze(1)  # batch 1 skill

            now_pro_embed = next_pro_rasch[:, now_step]  # batch d

            f1 = now_pro_embed.unsqueeze(1)  # batch 1 d
            f2 = skill_state  # batch skill d

            skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time  # batch skill
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), time_embed)  # batch skill d

            now_all_state = all_state  # batch d

            forget_now_all_state = now_all_state * self.all_forget(
                self.dropout(torch.cat([now_all_state, all_gap_embed], dim=-1)))

            effect_all_state = forget_now_all_state.unsqueeze(1).repeat(1, f2.shape[1], 1)

            skill_forget = torch.sigmoid(self.skill_forget(
                self.dropout(torch.cat([skill_state, skill_time_gap_embed, effect_all_state], dim=-1))))
            skill_forget = torch.masked_fill(skill_forget, now_pro2skill.transpose(-1, -2) == 0, 1)
            skill_state = skill_state * skill_forget

            # now_pro_skill_attn = self.do_attn(torch.cat([f1+f2,f2], dim=-1)) # batch skill 1
            now_pro_skill_attn = torch.matmul(f1, skill_state.transpose(-1, -2)) / f1.shape[-1]  # batch 1 skill

            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)

            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill

            now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)  # batch d

            all_attn = torch.sigmoid(self.predict_attn(
                self.dropout(torch.cat([now_need_state, forget_now_all_state, now_pro_embed], dim=-1))))

            now_need_state = torch.cat([(1 - all_attn) * now_need_state, all_attn * forget_now_all_state], dim=-1)

            last_skill_time = torch.masked_fill(last_skill_time, now_pro2skill.squeeze(1) == 1, now_step)

            now_ability = torch.sigmoid(self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1)))  # batch 1
            now_guess = next_pro_guess[:, now_step]  # batch 1
            now_divide = next_pro_divide[:, now_step]  # batch 1
            now_diff = F.embedding(now_pro, pro_diff)  # batch 1

            now_output = torch.sigmoid(5 * (now_ability - now_diff))

            now_output = now_output.squeeze(-1)

            res_p.append(now_output)

            now_X = next_X[:, now_step]  # batch d

            all_state = forget_now_all_state + torch.tanh(self.all_obtain(self.dropout(now_X))).squeeze(1)

            to_get = torch.tanh(self.now_obtain(self.dropout(now_X))).unsqueeze(1)  # batch 1 d

            f1 = to_get  # batch 1 d
            f2 = skill_state  # batch skill d

            now_pro_skill_attn = torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]  # batch 1 skill
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill

            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)

            skill_state = skill_state + now_get
            # skill_state

        P = torch.vstack(res_p).T

        out_dict = {
            "pred": P,
            "true": next_ans.float(),
            "contrast_loss": contrast_loss,
        }

        return out_dict
    
    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        loss = self.loss_fn(pred[mask], true[mask]) + out_dict["contrast_loss"]
        return loss, len(pred[mask]), true[mask].sum().item()