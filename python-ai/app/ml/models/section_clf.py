"""
PyTorch Model 3: SectionClassifierNet
Text CNN for Resume Section Classification.
Classifies a block of text into one of 8 resume sections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List


SECTION_LABELS = [
    "contact",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "other"
]

N_CLASSES = len(SECTION_LABELS)


class TextCNNBlock(nn.Module):
    """Single CNN block with a specific filter size (n-gram)."""

    def __init__(self, embed_dim: int, n_filters: int, filter_size: int):
        super(TextCNNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=n_filters,
            kernel_size=filter_size,
            padding=filter_size // 2
        )
        self.bn = nn.BatchNorm1d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim, seq_len)
        conv_out = F.relu(self.bn(self.conv(x)))    # (batch, n_filters, seq_len)
        pooled = F.adaptive_max_pool1d(conv_out, 1) # (batch, n_filters, 1)
        return pooled.squeeze(-1)                   # (batch, n_filters)


class SectionClassifierNet(nn.Module):
    """
    Multi-filter TextCNN for resume section classification.
    
    Uses 3 parallel CNN blocks with filter sizes [2, 3, 4] (bigrams, trigrams, 4-grams)
    then concatenates their outputs for classification.
    
    Input:  Token indices (batch, seq_len)
    Output: Section class probabilities (batch, 8)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 50,
        n_filters: int = 128,
        filter_sizes: List[int] = [2, 3, 4],
        dropout: float = 0.5,
        pretrained_embeddings: np.ndarray = None
    ):
        super(SectionClassifierNet, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )

        # Parallel CNN blocks for each n-gram size
        self.cnn_blocks = nn.ModuleList([
            TextCNNBlock(embed_dim, n_filters, fs)
            for fs in filter_sizes
        ])

        total_features = n_filters * len(filter_sizes)  # 128 * 3 = 384

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch, seq_len)
        Returns:
            logits: (batch, 8) — raw class scores
        """
        embedded = self.embedding(x)                     # (batch, seq, embed_dim)
        embedded = embedded.permute(0, 2, 1)             # (batch, embed_dim, seq)

        cnn_outputs = [block(embedded) for block in self.cnn_blocks]
        concat = torch.cat(cnn_outputs, dim=1)           # (batch, 384)

        return self.classifier(concat)                   # (batch, 8)

    def predict_section(self, x: torch.Tensor) -> str:
        """Inference: returns predicted section label."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            pred_idx = logits.argmax(dim=1).item()
            return SECTION_LABELS[pred_idx]

    def predict_all_sections(self, text_blocks: list, tokenizer) -> List[dict]:
        """
        Classify a list of text blocks into sections.
        Args:
            text_blocks: list of raw text strings
            tokenizer: tokenizer function (text -> token ids tensor)
        Returns:
            List of dicts with text, section label, confidence
        """
        self.eval()
        results = []
        with torch.no_grad():
            for block in text_blocks:
                tokens = tokenizer(block)
                logits = self.forward(tokens.unsqueeze(0))
                probs = F.softmax(logits, dim=1).squeeze()
                pred_idx = probs.argmax().item()
                results.append({
                    "text": block[:100],
                    "section": SECTION_LABELS[pred_idx],
                    "confidence": round(probs[pred_idx].item(), 4)
                })
        return results


def build_section_classifier(vocab_size: int = 10000) -> SectionClassifierNet:
    """Factory function to create a SectionClassifierNet."""
    return SectionClassifierNet(vocab_size=vocab_size)
