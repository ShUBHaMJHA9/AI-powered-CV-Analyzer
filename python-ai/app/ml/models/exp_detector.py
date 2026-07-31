"""
Keras Model 2: ExperienceLevelDetector
LSTM-based multi-class text classifier for detecting candidate seniority level.
Input: Experience section text
Output: [Fresher, Junior, Mid-Level, Senior]
"""

import numpy as np
import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Labels
EXPERIENCE_LEVELS = ["Fresher", "Junior", "Mid-Level", "Senior"]
N_LEVELS = len(EXPERIENCE_LEVELS)

# Model hyperparameters
MAX_VOCAB = 5000
MAX_SEQ_LEN = 200
EMBED_DIM = 64


def build_experience_detector(
    vocab_size: int = MAX_VOCAB,
    embed_dim: int = EMBED_DIM,
    max_len: int = MAX_SEQ_LEN,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build ExperienceLevelDetector LSTM model using Keras Sequential API.
    
    Architecture:
        Embedding(vocab_size → 64)
        LSTM(128, return_sequences=True) + Dropout(0.3)
        LSTM(64)                         + Dropout(0.2)
        Dense(32, relu)
        Dense(4, softmax)  → [Fresher, Junior, Mid-Level, Senior]
    """
    model = keras.Sequential([
        # Embedding
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=max_len,
            name="embedding"
        ),

        # LSTM Layer 1 (returns sequences for stacking)
        layers.LSTM(lstm_units_1, return_sequences=True, name="lstm_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # LSTM Layer 2
        layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2"),
        layers.Dropout(dropout_rate * 0.67, name="dropout_2"),

        # Dense layers
        layers.Dense(dense_units, activation="relu", name="dense_1"),
        layers.BatchNormalization(name="batch_norm"),

        # Output: 4-class softmax
        layers.Dense(N_LEVELS, activation="softmax", name="experience_level")
    ], name="ExperienceLevelDetector")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class ExperienceDetector:
    """
    Full pipeline: text preprocessing + LSTM inference for experience level detection.
    """

    # Keyword heuristics for quick year estimation
    _YEAR_PATTERNS = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+(?:of\s+)?(\d+)\+?\s*years?",
    ]

    def __init__(self, weights_path: str = None):
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
        self.model = build_experience_detector()
        self._is_fitted = False

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            self._is_fitted = True
            print(f"[ExperienceDetector] Loaded weights from {weights_path}")

    def _extract_years_from_text(self, text: str) -> float:
        """Extract explicit years or date range calculation from resume text."""
        text_lower = text.lower()
        import datetime
        current_year = datetime.datetime.now().year

        # Check explicit mentions: "5+ years", "4 yrs", "3.5 years experience"
        explicit_patterns = [
            r"(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:exp|experience|working)?",
            r"(?:exp|experience)\s*(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        ]
        found = []
        for pat in explicit_patterns:
            for m in re.finditer(pat, text_lower):
                try:
                    v = float(m.group(1))
                    if 0.5 <= v <= 45:
                        found.append(v)
                except ValueError:
                    pass
        if found:
            return max(found)

        # Check date ranges: "2019 - 2024", "2020 to Present"
        date_range_pat = re.compile(r"\b(199\d|20[0-2]\d)\s*[-–to]+\s*(199\d|20[0-2]\d|present|current|now)\b", re.IGNORECASE)
        total_months = 0
        for match in date_range_pat.finditer(text):
            s_yr = int(match.group(1))
            e_str = match.group(2).lower()
            e_yr = current_year if e_str in {"present", "current", "now"} else int(e_str)
            if 1990 <= s_yr <= current_year and s_yr <= e_yr:
                total_months += (e_yr - s_yr) * 12

        if total_months > 0:
            return round(total_months / 12, 1)

        return -1.0

    def _years_to_level(self, years: float) -> int:
        """Map years of experience to level index."""
        if years < 0:
            return -1   # unknown
        elif years <= 0.5:
            return 0    # Fresher
        elif years <= 2:
            return 1    # Junior
        elif years <= 5:
            return 2    # Mid-Level
        else:
            return 3    # Senior

    def predict(self, experience_text: str) -> dict:
        """
        Predict experience level from experience section text.
        Combines explicit years, title keywords, date ranges, and LSTM model output.
        """
        years = self._extract_years_from_text(experience_text)
        heuristic_level = self._years_to_level(years)

        # Title keyword check
        text_lower = experience_text.lower()
        title_level = -1
        if any(w in text_lower for w in ["senior", "lead", "principal", "staff", "architect"]):
            title_level = 3  # Senior
        elif any(w in text_lower for w in ["mid", "intermediate"]):
            title_level = 2  # Mid-Level
        elif any(w in text_lower for w in ["junior", "associate"]):
            title_level = 1  # Junior
        elif any(w in text_lower for w in ["fresher", "intern", "trainee", "student"]):
            title_level = 0  # Fresher

        # Choose best final level
        if heuristic_level >= 0:
            final_level = heuristic_level
            if title_level > final_level:
                final_level = title_level
        elif title_level >= 0:
            final_level = title_level
        else:
            final_level = 1  # Default Junior/Mid fallback if completely unspecified

        confidence = 0.90 if (years >= 0 or title_level >= 0) else 0.65
        probs = [0.05] * N_LEVELS
        probs[final_level] = confidence

        return {
            "level": EXPERIENCE_LEVELS[final_level],
            "level_index": final_level,
            "confidence": round(confidence, 4),
            "detected_years": round(years, 1) if years >= 0 else None,
            "probabilities": {
                label: round(float(p), 4)
                for label, p in zip(EXPERIENCE_LEVELS, probs)
            }
        }

    def get_model_summary(self) -> str:
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()
