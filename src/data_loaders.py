from torch.utils.data import Dataset
import os
from utils.augment_seq import (
    preprocess_qr,
    preprocess_qsr,
    augment_kt_seqs,
    replace_only,
    counter_kt_seqs,
)
import torch
from collections import defaultdict


class SimCLRDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
        mask_prob: float,
        crop_prob: float,
        permute_prob: float,
        replace_prob: float,
        negative_prob: float,
        eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        self.eval_mode = eval_mode

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.easier_skills = self.ds.easier_skills
        self.harder_skills = self.ds.harder_skills

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]

        if self.eval_mode:
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "attention_mask": attention_mask,
            }

        else:
            q_seq_list = original_data["questions"].tolist()
            s_seq_list = original_data["skills"].tolist()
            r_seq_list = original_data["responses"].tolist()

            t1 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
            )

            t2 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index + 1,
            )

            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, negative_r_seq, attention_mask_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2, _, attention_mask_2 = t2

            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.long)
            aug_q_seq_2 = torch.tensor(aug_q_seq_2, dtype=torch.long)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.long)
            aug_s_seq_2 = torch.tensor(aug_s_seq_2, dtype=torch.long)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.long)
            aug_r_seq_2 = torch.tensor(aug_r_seq_2, dtype=torch.long)
            negative_r_seq = torch.tensor(negative_r_seq, dtype=torch.long)
            attention_mask_1 = torch.tensor(attention_mask_1, dtype=torch.long)
            attention_mask_2 = torch.tensor(attention_mask_2, dtype=torch.long)

            ret = {
                "questions": (aug_q_seq_1, aug_q_seq_2, q_seq),
                "skills": (aug_s_seq_1, aug_s_seq_2, s_seq),
                "responses": (aug_r_seq_1, aug_r_seq_2, r_seq, negative_r_seq),
                "attention_mask": (attention_mask_1, attention_mask_2, attention_mask),
            }
            return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)


class CounterDatasetWrapper(Dataset):
    def __init__(
        self,
        ds: Dataset,
        seq_len: int,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len

        self.num_questions = self.ds.num_questions
        self.num_skills = self.ds.num_skills
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.skill_difficulty = self.ds.skill_difficulty

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        attention_mask = original_data["attention_mask"]
        q_seq_list = original_data["questions"].tolist()
        s_seq_list = original_data["skills"].tolist()
        r_seq_list = original_data["responses"].tolist()

        t = counter_kt_seqs(
            q_seq_list,
            s_seq_list,
            r_seq_list,
            self.skill_difficulty,
            seed=index,
        )

        counter_attention_mask = t

        counter_attention_mask = torch.tensor(counter_attention_mask, dtype=torch.long)

        ret = {
            "questions": q_seq,
            "skills": s_seq,
            "responses": r_seq,
            "attention_mask": (counter_attention_mask, attention_mask),
        }
        return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)


class MostRecentQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.times = [
            u_df["timestamp"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(self.skills, self.responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }

        # print(f'diff = {self.skill_difficulty}')
        # import sys
        # sys.exit()

        ordered_skills = [
            item[0] for item in sorted(self.skill_difficulty.items(), key=lambda x: x[1])
        ]

        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            self.padded_q[i, -len(q) :] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s) :] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r) :] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, -len(s) :] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class MostEarlyQuestionSkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["item_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.skills = [
            u_df["skill_id"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.times = [
            u_df["timestamp"].values[-self.seq_len :]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values[: self.seq_len]
            for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.len = len(self.questions)

        self.padded_q = torch.zeros(
            (len(self.questions), self.seq_len), dtype=torch.long
        )
        self.padded_s = torch.zeros((len(self.skills), self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((len(self.times), self.seq_len), dtype=torch.long)
        self.padded_r = torch.full(
            (len(self.responses), self.seq_len), -1, dtype=torch.long
        )
        self.attention_mask = torch.zeros(
            (len(self.skills), self.seq_len), dtype=torch.long
        )

        for i, elem in enumerate(zip(self.questions, self.skills, self.responses, self.times)):
            q, s, r, t = elem
            self.padded_q[i, : len(q)] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, : len(r)] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class SkillDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        self.questions = [
            u_df["skill_id"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.responses = [
            u_df["correct"].values for _, u_df in self.df.groupby("user_id")
        ]
        self.lengths = [
            len(u_df["skill_id"].values) for _, u_df in self.df.groupby("user_id")
        ]

        cnt = 0
        for interactions in self.questions:
            cnt += len(interactions)
        self.num_interactions = cnt

        self.questions, self.responses = preprocess_qr(
            self.questions, self.responses, self.seq_len
        )
        self.len = len(self.questions)

    def __getitem__(self, index):
        return {"questions": self.questions[index], "responses": self.responses[index]}

    def __len__(self):
        return self.len


class MostRecentQuestionSkillStackedDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        # Get raw sequences for each user
        raw_questions = []
        raw_skills = []
        raw_responses = []
        raw_times = []

        # Process each user's sequence and split into chunks if needed
        for _, u_df in self.df.groupby("user_id"):
            q_seq = u_df["item_id"].values
            s_seq = u_df["skill_id"].values
            r_seq = u_df["correct"].values
            t_seq = u_df["timestamp"].values

            # If sequence length > seq_len, split into multiple sequences
            if len(q_seq) > seq_len:
                for start_idx in range(0, len(q_seq) - seq_len + 1, seq_len):
                    end_idx = start_idx + seq_len
                    raw_questions.append(q_seq[start_idx:end_idx])
                    raw_skills.append(s_seq[start_idx:end_idx])
                    raw_responses.append(r_seq[start_idx:end_idx])
                    raw_times.append(t_seq[start_idx:end_idx])
            else:
                # For shorter sequences, keep as is
                raw_questions.append(q_seq)
                raw_skills.append(s_seq)
                raw_responses.append(r_seq)
                raw_times.append(t_seq)

        # Calculate skill difficulties
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(raw_skills, raw_responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }

        ordered_skills = [
            item[0] for item in sorted(self.skill_difficulty.items(), key=lambda x: x[1])
        ]

        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        # Count total interactions
        self.num_interactions = sum(len(q) for q in raw_questions)
        self.len = len(raw_questions)

        # Create padded tensors
        self.padded_q = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_s = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_r = torch.full((self.len, self.seq_len), -1, dtype=torch.long)
        self.attention_mask = torch.zeros((self.len, self.seq_len), dtype=torch.long)

        # Fill padded tensors
        for i, (q, s, r, t) in enumerate(zip(raw_questions, raw_skills, raw_responses, raw_times)):
            self.padded_q[i, -len(q):] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, -len(s):] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, -len(r):] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, -len(t):] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, -len(s):] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len


class MostEarlyQuestionSkillStackedDataset(Dataset):
    def __init__(self, df, seq_len, num_skills, num_questions):
        self.df = df
        self.seq_len = seq_len
        self.num_skills = num_skills
        self.num_questions = num_questions

        # Get raw sequences for each user
        raw_questions = []
        raw_skills = []
        raw_responses = []
        raw_times = []

        # Process each user's sequence and split into chunks if needed
        for _, u_df in self.df.groupby("user_id"):
            q_seq = u_df["item_id"].values
            s_seq = u_df["skill_id"].values
            r_seq = u_df["correct"].values
            t_seq = u_df["timestamp"].values

            # If sequence length > seq_len, split into multiple sequences
            # For early dataset, we start from the beginning of each user's sequence
            if len(q_seq) > seq_len:
                for start_idx in range(0, len(q_seq) - seq_len + 1, seq_len):
                    end_idx = start_idx + seq_len
                    raw_questions.append(q_seq[start_idx:end_idx])
                    raw_skills.append(s_seq[start_idx:end_idx])
                    raw_responses.append(r_seq[start_idx:end_idx])
                    raw_times.append(t_seq[start_idx:end_idx])
            else:
                # For shorter sequences, keep as is
                raw_questions.append(q_seq)
                raw_skills.append(s_seq)
                raw_responses.append(r_seq)
                raw_times.append(t_seq)

        # Calculate skill difficulties
        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(raw_skills, raw_responses):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += r
                skill_count[s] += 1

        self.skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }

        ordered_skills = [
            item[0] for item in sorted(self.skill_difficulty.items(), key=lambda x: x[1])
        ]

        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        # Count total interactions
        self.num_interactions = sum(len(q) for q in raw_questions)
        self.len = len(raw_questions)

        # Create padded tensors
        self.padded_q = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_s = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_t = torch.zeros((self.len, self.seq_len), dtype=torch.long)
        self.padded_r = torch.full((self.len, self.seq_len), -1, dtype=torch.long)
        self.attention_mask = torch.zeros((self.len, self.seq_len), dtype=torch.long)

        # Fill padded tensors - for early dataset, we place data at the beginning of tensors
        for i, (q, s, r, t) in enumerate(zip(raw_questions, raw_skills, raw_responses, raw_times)):
            self.padded_q[i, :len(q)] = torch.tensor(q, dtype=torch.long)
            self.padded_s[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            self.padded_r[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            self.padded_t[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            self.attention_mask[i, :len(s)] = torch.ones(len(s), dtype=torch.long)

    def __getitem__(self, index):
        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "times": self.padded_t[index],
            "attention_mask": self.attention_mask[index],
        }

    def __len__(self):
        return self.len
