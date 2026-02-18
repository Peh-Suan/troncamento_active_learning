# import torch.nn as nn
# import torch

# class MisalignmentDetector(nn.Module):
#     def __init__(self, dim=1024, n_phones=50, phone_emb_dim=32):
#         super().__init__()

#         self.phone_emb = nn.Embedding(n_phones, phone_emb_dim)

#         self.net = nn.Sequential(
#             nn.Linear(dim + phone_emb_dim * 2 + 1, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x, phone_ids, silver_labels):
#         phone_vecs = self.phone_emb(phone_ids).view(x.size(0), -1)
#         x = torch.cat([x, phone_vecs, silver_labels.unsqueeze(-1)], dim=-1)
#         # x = torch.cat([x, phone_vecs], dim=-1)
#         return self.net(x).squeeze(-1)

import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, dim=1024, hidden=256, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # -> score per frame
        )

    def forward(self, x, mask):
        """
        x:    (B, T, D)
        mask: (B, T) bool, True for valid frames
        returns pooled: (B, D)
        """
        scores = self.scorer(x).squeeze(-1)  # (B, T)

        # Put -inf on padded positions so softmax ignores them
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, T)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, attn

class MisalignmentDetector(nn.Module):
    def __init__(self, dim=1024, n_phones=50, phone_emb_dim=32):
        super().__init__()
        self.phone_emb = nn.Embedding(n_phones, phone_emb_dim)
        self.pool = AttentionPooling(dim=dim, hidden=256, dropout=0.1)

        self.shared = nn.Sequential(
            nn.Linear(dim + phone_emb_dim * 2 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_bad = nn.Linear(128, 1)  # misaligned?
        self.head_e   = nn.Linear(128, 1)  # /e/?

    def forward(self, hidden_seq, mask, phone_ids, silver_labels):
        pooled, attn = self.pool(hidden_seq, mask)  # (B,1024)
        phone_vecs = self.phone_emb(phone_ids).view(hidden_seq.size(0), -1)  # (B,2*emb)

        x = torch.cat([pooled, phone_vecs, silver_labels.unsqueeze(-1)], dim=-1)
        h = self.shared(x)  # (B,128)

        logit_bad = self.head_bad(h).squeeze(-1)  # (B,)
        logit_e   = self.head_e(h).squeeze(-1)    # (B,)
        return logit_e, logit_bad, attn
