"""
Training Script for Keras Models:
 - ATSClassifierModel (Bidirectional LSTM)
 - ExperienceLevelDetector (LSTM)
Run this ONCE to train and save model weights.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_ats_keywords() -> list:
    ats_path = DATA_DIR / "ats_keywords.json"
    if ats_path.exists():
        return json.loads(ats_path.read_text())
    # Fallback minimal list
    return [
        "python", "machine learning", "deep learning", "sql", "javascript",
        "react", "node.js", "aws", "docker", "kubernetes", "tensorflow",
        "pytorch", "keras", "scikit-learn", "pandas", "numpy", "nlp",
        "git", "github", "agile", "scrum", "rest api", "microservices",
        "postgresql", "mongodb", "data analysis", "statistics", "excel",
        "power bi", "tableau", "communication", "leadership", "teamwork"
    ] * 15  # pad to 500+


# ─── Train ATSClassifierModel ─────────────────────────────────────────────────

def train_ats_classifier(epochs: int = 30, batch_size: int = 128):
    from app.ml.models.ats_model import ATSClassifier, build_ats_model
    from app.ml.training.synthetic_data import generate_ats_data
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    print("\n[1/2] Training ATSClassifierModel (Keras Bidirectional LSTM)...")

    ats_keywords = load_ats_keywords()[:500]
    X, y = generate_ats_data(ats_keywords, n_samples=5000)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = build_ats_model(vocab_size=len(ats_keywords))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(WEIGHTS_DIR / "ats_classifier.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    final_val_auc = max(history.history["val_auc"])
    print(f"  ✅ ATSClassifierModel saved (best val_AUC={final_val_auc:.4f})")


# ─── Train ExperienceLevelDetector ───────────────────────────────────────────

def train_experience_detector(epochs: int = 20, batch_size: int = 32):
    from app.ml.models.exp_detector import (
        ExperienceDetector, build_experience_detector,
        MAX_VOCAB, MAX_SEQ_LEN
    )
    from app.ml.training.synthetic_data import generate_experience_texts
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import pickle

    print("\n[2/2] Training ExperienceLevelDetector (Keras LSTM)...")

    texts, labels = generate_experience_texts()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # Fit tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_raw)

    # Save tokenizer for inference
    with open(WEIGHTS_DIR / "exp_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Prepare sequences
    X_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train_raw),
        maxlen=MAX_SEQ_LEN, padding="post"
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(X_val_raw),
        maxlen=MAX_SEQ_LEN, padding="post"
    )
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    model = build_experience_detector(vocab_size=vocab_size)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(WEIGHTS_DIR / "exp_detector.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    final_val_acc = max(history.history["val_accuracy"])
    print(f"  ✅ ExperienceLevelDetector saved (best val_acc={final_val_acc:.4f})")


if __name__ == "__main__":
    train_ats_classifier()
    train_experience_detector()
    print("\n🎉 All Keras models trained and saved to python-ai/app/ml/weights/")
