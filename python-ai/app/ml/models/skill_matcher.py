"""
PyTorch Model 1: SkillMatcherNet
Siamese Bidirectional LSTM Network for Semantic Skill Matching
Compares user skills vs. target role skills and returns similarity score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SkillMatcherNet(nn.Module):
    """
    Siamese BiLSTM network that takes two skill sequences and computes
    semantic similarity score between them.
    
    Architecture:
        - Shared embedding layer (GloVe 50d or random init)
        - Shared BiLSTM encoder (128 hidden units)
        - Mean pooling
        - Cosine similarity output
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: np.ndarray = None
    ):
        super(SkillMatcherNet, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Shared Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )

        # Shared BiLSTM Encoder (used for BOTH input sequences)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Projection layer to reduce dimension
        self.projection = nn.Linear(hidden_dim * 2, 128)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a skill sequence into a fixed-size vector."""
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))          # (batch, seq, embed_dim)
        lstm_out, _ = self.bilstm(embedded)                  # (batch, seq, hidden*2)
        lstm_out = self.layer_norm(lstm_out)

        # Mean pooling over sequence (ignores padding with mask)
        mask = (x != 0).float().unsqueeze(-1)               # (batch, seq, 1)
        pooled = (lstm_out * mask).sum(1) / mask.sum(1).clamp(min=1)  # (batch, hidden*2)

        return F.relu(self.projection(self.dropout(pooled))) # (batch, 128)

    def forward(self, skills_a: torch.Tensor, skills_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            skills_a: User skill tokens (batch, seq_len)
            skills_b: Role requirement tokens (batch, seq_len)
        Returns:
            similarity: Cosine similarity scores (batch,) in range [0, 1]
        """
        vec_a = self.encode(skills_a)   # (batch, 128)
        vec_b = self.encode(skills_b)   # (batch, 128)

        # Cosine similarity → scale to [0, 1]
        similarity = F.cosine_similarity(vec_a, vec_b, dim=1)
        return (similarity + 1) / 2     # map [-1,1] → [0,1]

    def predict_score(self, skills_a: torch.Tensor, skills_b: torch.Tensor) -> float:
        """Inference: returns skill match percentage (0-100)."""
        self.eval()
        with torch.no_grad():
            score = self.forward(skills_a, skills_b)
            return round(score.mean().item() * 100, 2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the Siamese network.
    Pulls similar pairs together, pushes dissimilar pairs apart.
    """

    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        label: torch.Tensor  # 1 = similar, 0 = dissimilar
    ) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = label * euclidean_distance.pow(2) + \
               (1 - label) * F.relu(self.margin - euclidean_distance).pow(2)
        return loss.mean()


def build_skill_matcher(vocab_size: int = 10000, embed_dim: int = 50) -> SkillMatcherNet:
    """Factory function to create a SkillMatcherNet."""
    return SkillMatcherNet(vocab_size=vocab_size, embed_dim=embed_dim)
