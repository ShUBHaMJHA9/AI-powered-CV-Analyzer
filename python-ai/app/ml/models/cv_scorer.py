"""
PyTorch Model 2: CVScoringNet
Deep Feedforward Neural Network for CV Quality Scoring.
Takes 35 engineered features from a CV and outputs quality score (0-100).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Feature index mapping (must match feature_extractor.py) ──────────────────
FEATURE_NAMES = [
    # Contact Info (3)
    "has_email", "has_phone", "has_linkedin_url",
    # Sections (8)
    "has_summary", "has_experience", "has_education", "has_skills",
    "has_projects", "has_certifications", "has_github_url", "section_count",
    # Content Quality (10)
    "word_count_norm",          # word_count / 800 (normalized)
    "action_verb_count_norm",   # action_verbs / 20
    "quantified_achievements",  # count of numbers/metrics in bullet points
    "bullet_point_count_norm",  # bullets / 15
    "avg_sentence_length_norm", # avg sentence length / 20
    "passive_voice_ratio",      # passive sentences / total (lower is better)
    "unique_word_ratio",        # unique words / total words
    "readability_score_norm",   # Flesch reading ease / 100
    "technical_term_density",   # tech words / total words
    "professional_tone_score",  # rule-based tone detection
    # Skills (5)
    "skill_count_norm",         # skills / 20
    "skill_category_diversity", # unique categories / 8
    "has_programming_skills",
    "has_data_skills",
    "has_cloud_skills",
    # Experience (5)
    "years_experience_norm",    # years / 10
    "job_count_norm",           # jobs / 5
    "avg_tenure_norm",          # avg months per job / 24
    "has_leadership_keywords",
    "experience_growth",        # seniority progression detected
    # Education (4)
    "education_level",          # 0=none,0.25=diploma,0.5=bachelor,0.75=master,1=phd
    "has_cgpa",
    "cgpa_norm",                # cgpa / 10.0
    "education_relevance",      # CS/IT/relevant field detected
]

N_FEATURES = len(FEATURE_NAMES)  # 35


class CVScoringNet(nn.Module):
    """
    4-layer feedforward neural network for CV quality scoring.
    
    Input:  35 engineered features (numpy array → torch tensor)
    Output: CV quality score in range [0, 100]
    
    Architecture:
        Linear(35→128) + BatchNorm + ReLU + Dropout(0.3)
        Linear(128→64) + BatchNorm + ReLU + Dropout(0.2)
        Linear(64→32)  + ReLU
        Linear(32→1)   + Sigmoid × 100
    """

    def __init__(self, n_features: int = N_FEATURES, dropout1: float = 0.3, dropout2: float = 0.2):
        super(CVScoringNet, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout1),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout2),

            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (batch_size, 35)
        Returns:
            scores: Quality scores (batch_size, 1) in [0, 1]
        """
        return self.network(x)

    def predict_score(self, features: np.ndarray) -> float:
        """
        Inference method: accepts numpy array, returns score 0-100.
        Args:
            features: shape (35,) or (batch, 35)
        Returns:
            score: float in [0, 100]
        """
        self.eval()
        with torch.no_grad():
            if features.ndim == 1:
                features = features.reshape(1, -1)
            x = torch.tensor(features, dtype=torch.float32)
            score = self.forward(x)
            return round(score.squeeze().item() * 100, 2)

    def predict_with_breakdown(self, features: np.ndarray) -> dict:
        """
        Returns score + contribution of each feature group for explainability.
        """
        score = self.predict_score(features)

        # Map feature groups to score contribution (proportional to feature values)
        feature_array = features.flatten()
        breakdown = {
            "contact_info":    round(np.mean(feature_array[0:3]) * 10, 2),
            "section_quality": round(np.mean(feature_array[3:11]) * 12, 2),
            "content_depth":   round(np.mean(feature_array[11:21]) * 30, 2),
            "skills_coverage": round(np.mean(feature_array[21:26]) * 25, 2),
            "experience":      round(np.mean(feature_array[26:31]) * 20, 2),
            "education":       round(np.mean(feature_array[31:35]) * 10, 2),
        }
        return {"overall": score, "breakdown": breakdown}


def build_cv_scorer(n_features: int = N_FEATURES) -> CVScoringNet:
    """Factory function to create a CVScoringNet."""
    return CVScoringNet(n_features=n_features)
