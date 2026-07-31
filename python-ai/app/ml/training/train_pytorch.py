"""
Training Script for PyTorch Models:
 - CVScoringNet (Feedforward NN)
 - SectionClassifierNet (TextCNN)
 - SkillMatcherNet (Siamese BiLSTM)
Run this script ONCE to train and save model weights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Training] Using device: {DEVICE}")


# ─── Train CVScoringNet ───────────────────────────────────────────────────────

def train_cv_scorer(epochs: int = 50, batch_size: int = 256):
    from app.ml.models.cv_scorer import CVScoringNet, N_FEATURES
    from app.ml.training.synthetic_data import generate_cv_features

    print("\n[1/3] Training CVScoringNet (PyTorch Feedforward NN)...")

    X, y = generate_cv_features(n_samples=10000)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CVScoringNet(n_features=N_FEATURES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "cv_scorer.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    print(f"  ✅ CVScoringNet saved → weights/cv_scorer.pt (best val_loss={best_val_loss:.4f})")


# ─── Train SectionClassifierNet ───────────────────────────────────────────────

def train_section_classifier(epochs: int = 30, batch_size: int = 64):
    from app.ml.models.section_clf import SectionClassifierNet
    from app.ml.training.synthetic_data import generate_section_data

    print("\n[2/3] Training SectionClassifierNet (PyTorch TextCNN)...")

    texts, labels = generate_section_data(n_per_class=500)

    # Simple character-level tokenizer
    from collections import Counter
    all_words = " ".join(texts).lower().split()
    vocab = {w: i+1 for i, (w, _) in enumerate(Counter(all_words).most_common(9999))}
    vocab["<PAD>"] = 0
    torch.save(vocab, WEIGHTS_DIR / "section_vocab.pt")

    MAX_LEN = 50

    def tokenize(text, max_len=MAX_LEN):
        tokens = [vocab.get(w.lower(), 0) for w in text.split()]
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return tokens

    X = torch.tensor([tokenize(t) for t in texts], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SectionClassifierNet(vocab_size=len(vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "section_clf.pt")

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Val Accuracy: {val_acc:.4f}")

    print(f"  ✅ SectionClassifierNet saved → weights/section_clf.pt (best acc={best_val_acc:.4f})")


# ─── Train SkillMatcherNet ────────────────────────────────────────────────────

def train_skill_matcher(epochs: int = 40, batch_size: int = 128):
    from app.ml.models.skill_matcher import SkillMatcherNet, ContrastiveLoss

    print("\n[3/3] Training SkillMatcherNet (PyTorch Siamese BiLSTM)...")

    # Skill pair categories for contrastive training
    similar_pairs = [
        ("python machine learning", "scikit learn tensorflow keras"),
        ("react javascript frontend", "vuejs angular typescript"),
        ("sql postgresql database", "mysql oracle nosql mongodb"),
        ("aws cloud computing", "azure gcp devops kubernetes"),
        ("nlp spacy nltk bert", "natural language processing transformers"),
    ]
    dissimilar_pairs = [
        ("python machine learning", "project management scrum agile"),
        ("react javascript frontend", "sql database administration"),
        ("aws cloud computing", "graphic design photoshop illustrator"),
        ("nlp text analysis", "civil engineering autocad"),
        ("deep learning neural network", "accounting finance excel"),
    ]

    # Build simple word vocab
    all_text = " ".join([t for pair in similar_pairs + dissimilar_pairs for t in pair])
    words = list(set(all_text.split()))
    vocab = {w: i+1 for i, w in enumerate(words)}
    torch.save(vocab, WEIGHTS_DIR / "skill_vocab.pt")

    MAX_LEN = 10

    def encode(text):
        tokens = [vocab.get(w, 0) for w in text.split()]
        tokens = tokens[:MAX_LEN] + [0] * max(0, MAX_LEN - len(tokens))
        return tokens

    # Augment dataset
    pairs_a, pairs_b, labels = [], [], []
    for _ in range(500):
        a, b = random.choice(similar_pairs)
        pairs_a.append(encode(a)); pairs_b.append(encode(b)); labels.append(1.0)
    for _ in range(500):
        a, b = random.choice(dissimilar_pairs)
        pairs_a.append(encode(a)); pairs_b.append(encode(b)); labels.append(0.0)

    import random
    combined = list(zip(pairs_a, pairs_b, labels))
    random.shuffle(combined)
    pairs_a, pairs_b, labels = zip(*combined)

    Xa = torch.tensor(list(pairs_a), dtype=torch.long)
    Xb = torch.tensor(list(pairs_b), dtype=torch.long)
    y = torch.tensor(list(labels), dtype=torch.float32)

    dataset = TensorDataset(Xa, Xb, y)
    train_ds, val_ds = random_split(dataset, [900, 100])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SkillMatcherNet(vocab_size=len(vocab)+1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xa, xb, yb in train_loader:
            xa, xb, yb = xa.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            similarity = model(xa, xb)
            loss = criterion(similarity, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), WEIGHTS_DIR / "skill_matcher.pt")
    print(f"  ✅ SkillMatcherNet saved → weights/skill_matcher.pt")


if __name__ == "__main__":
    import random
    train_cv_scorer()
    train_section_classifier()
    train_skill_matcher()
    print("\n🎉 All PyTorch models trained and saved to python-ai/app/ml/weights/")
