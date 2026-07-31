"""
Keras Model 1: ATSClassifierModel
Bidirectional LSTM for ATS (Applicant Tracking System) Compatibility Scoring.
Predicts the probability that a resume will pass ATS filtering.
"""

import numpy as np
import os

# Use TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 500 ATS keywords tracked (keyword presence vector length)
ATS_VOCAB_SIZE = 500


def build_ats_model(
    vocab_size: int = ATS_VOCAB_SIZE,
    embed_dim: int = 32,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Build the ATSClassifierModel using Keras functional API.
    
    Architecture:
        Embedding(500 → 32)
        Bidirectional LSTM(64) + Dropout
        Bidirectional LSTM(32) + Dropout
        Dense(64, relu)
        Dense(1, sigmoid) → ATS pass probability

    Args:
        vocab_size: Number of unique ATS keywords tracked
        embed_dim: Embedding dimension for keyword vectors
        lstm_units_1: Units in first BiLSTM layer
        lstm_units_2: Units in second BiLSTM layer
        dense_units: Dense layer size
        dropout_rate: Dropout probability

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(vocab_size,), name="keyword_presence_vector")

    # Reshape for LSTM: (batch, seq_len=1, vocab_size)
    x = layers.Reshape((1, vocab_size))(inputs)

    # Reproject to lower dimension first
    x = layers.Dense(embed_dim, activation="relu", name="embed_dense")(x)

    # BiLSTM Layer 1
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True, name="bilstm_1")
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # BiLSTM Layer 2
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_2, return_sequences=False, name="bilstm_2")
    )(x)
    x = layers.Dropout(dropout_rate * 0.67)(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output
    output = layers.Dense(1, activation="sigmoid", name="ats_score")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="ATSClassifierModel")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


class ATSClassifier:
    """
    Wrapper class for ATS classification with keyword extraction utilities.
    """

    def __init__(self, ats_keywords: list, weights_path: str = None):
        """
        Args:
            ats_keywords: List of 500 ATS keyword strings (ordered)
            weights_path: Path to saved model weights (.h5 or .keras)
        """
        self.ats_keywords = ats_keywords
        self.keyword_to_idx = {kw.lower(): i for i, kw in enumerate(ats_keywords)}
        self.model = build_ats_model(vocab_size=len(ats_keywords))

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"[ATSClassifier] Loaded weights from {weights_path}")

    def text_to_keyword_vector(self, resume_text: str) -> np.ndarray:
        """
        Convert resume text to a binary keyword presence vector.
        Args:
            resume_text: Full resume text string
        Returns:
            vector: np.ndarray of shape (500,) with 1s where keywords appear
        """
        text_lower = resume_text.lower()
        vector = np.zeros(len(self.ats_keywords), dtype=np.float32)

        for kw, idx in self.keyword_to_idx.items():
            if kw in text_lower:
                vector[idx] = 1.0

        return vector

    def predict(self, resume_text: str) -> dict:
        """
        Predict ATS compatibility for a resume.
        Args:
            resume_text: Raw resume text
        Returns:
            dict with score, pass/fail, matched keywords, missing keywords
        """
        keyword_vector = self.text_to_keyword_vector(resume_text)
        input_tensor = keyword_vector.reshape(1, -1)

        ats_prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
        ats_score = round(ats_prob * 100, 2)

        matched = [self.ats_keywords[i] for i, v in enumerate(keyword_vector) if v == 1.0]
        missing = [self.ats_keywords[i] for i, v in enumerate(keyword_vector) if v == 0.0]

        return {
            "ats_score": ats_score,
            "ats_probability": round(ats_prob, 4),
            "ats_pass": ats_prob >= 0.5,
            "keywords_matched": matched[:20],
            "keywords_missing": missing[:20],
            "match_rate": round(len(matched) / len(self.ats_keywords) * 100, 2)
        }

    def get_model_summary(self) -> str:
        """Return model architecture summary as string."""
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()
