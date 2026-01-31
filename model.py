import torch.nn as nn
import torch

class MisalignmentDetector(nn.Module):
    def __init__(self, dim=1024, n_phones=50, phone_emb_dim=32):
        super().__init__()

        self.phone_emb = nn.Embedding(n_phones, phone_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(dim + phone_emb_dim * 2 + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, phone_ids, silver_labels):
        phone_vecs = self.phone_emb(phone_ids).view(x.size(0), -1)
        x = torch.cat([x, phone_vecs, silver_labels.unsqueeze(-1)], dim=-1)
        # x = torch.cat([x, phone_vecs], dim=-1)
        return self.net(x).squeeze(-1)
