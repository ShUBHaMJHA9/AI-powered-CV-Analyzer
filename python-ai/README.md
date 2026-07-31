# 🐍 SMARRTIF AI — Python FastAPI Microservice & AI Engine

The core AI engine of **SMARRTIF AI** is a high-performance **FastAPI** microservice. It orchestrates **5 custom Deep Learning & NLP models** built using **PyTorch**, **Keras/TensorFlow**, **spaCy**, and **NLTK** to evaluate CVs, GitHub repositories, and LinkedIn profiles.

> **Made with ❤️ by [Shubham Kumar Jha](mailto:shubhamjha22088@gmail.com)**  
> ✉️ **Contact:** `shubhamjha22088@gmail.com`

---

## 🧠 Deep Learning & NLP Model Architecture

```
                                  ┌───────────────────────────────┐
                                  │       Raw CV Text File        │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │      NLPEngine (spaCy C)      │
                                  │  NER • Skills • Dates • Quality│
                                  └───────────────┬───────────────┘
                                                  │
        ┌───────────────────┬─────────────────────┼─────────────────────┬───────────────────┐
        │                   │                     │                     │                   │
        ▼                   ▼                     ▼                     ▼                   ▼
┌───────────────┐   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐   ┌───────────────┐
│ CVScoringNet  │   │SkillMatcherNet│     │ ATSClassifier │     │  Experience   │   │  GitHub &     │
│ (PyTorch FFNN)│   │(Siamese BiLSTM│     │ (Keras BiLSTM)│     │ Detector (LSTM│   │ LinkedIn API  │
│  30 Pts Max   │   │  25 Pts Max   │     │  15 Pts Max   │     │  20 Pts Max   │   │  10 Pts Each  │
└───────┬───────┘   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘   └───────┬───────┘
        │                   │                     │                     │                   │
        └───────────────────┴─────────────────────┼─────────────────────┴───────────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ ScoringEngine Master Aggregator│
                                  │   Overall Score & Grade Report│
                                  └───────────────────────────────┘
```

---

## 🔬 Detailed Model Specifications

### 1. CVScoringNet (PyTorch Feedforward NN)
- **Input:** 35 engineered feature vector constructed by `FeatureExtractor` (contact completeness, section counts, word density, action verbs, readability score, skill category diversity, tenure, education level).
- **Architecture:** 4-Layer Linear Network (`35 → 128 → 64 → 32 → 1`) with BatchNorm1d, ReLU activations, and Dropout (`0.3`, `0.2`).
- **Output:** Overall CV quality score (0–100 scaled to **30 Pts Max**).

### 2. SkillMatcherNet (PyTorch Siamese BiLSTM)
- **Input:** Tokenized candidate skills vs required role skills from 12+ pre-defined role templates.
- **Architecture:** Dual-branch Siamese BiLSTM with Shared Embedding (`500 → 64`) and Cosine Similarity metric.
- **Output:** Skill alignment score (0–1 scaled to **25 Pts Max**) + missing skills gap list.

### 3. ATSClassifierModel (Keras Bidirectional LSTM)
- **Input:** 500-dimension binary keyword presence vector constructed from tracked ATS vocabulary.
- **Architecture:** `Input(500) → Dense(32) → BiLSTM(64) → BiLSTM(32) → Dense(64) → Sigmoid(1)`.
- **Output:** ATS Pass Probability (0–100% scaled to **15 Pts Max**) + top missing ATS keywords.

### 4. ExperienceLevelDetector (Keras LSTM + Hybrid Regex Engine)
- **Input:** Full experience section text.
- **Architecture:** `Embedding(5000 → 64) → Stacked LSTM(128, 64) → Dense(32) → Softmax(4)`.
- **Hybrid Parser:** Evaluates explicit year mentions (`"5+ years of experience"`), date ranges (`"2019-2024"`), and seniority title keywords (`Senior`, `Lead`, `Junior`, `Intern`).
- **Output:** Seniority classification (`Fresher`, `Junior`, `Mid-Level`, `Senior`) + detected years (scaled to **20 Pts Max**).

### 5. GitHub & LinkedIn Intelligence Engines
- **GitHub Engine:** Parses clean handles from URLs/mentions → Queries GitHub REST API → Calculates public repos count, star count ⭐, followers count 👥, forks count, and top language distribution (%) (**10 Pts Max**).
- **LinkedIn Engine:** Scrapes OpenGraph meta, JSON-LD schema, and user inputs → Evaluates headline match, skills alignment, college standing (`IIT`, `NIT`, `BITS`, `Stanford`, `Degree`), certifications (`AWS`, `PMP`, `TensorFlow`), and URL validity (**10 Pts Max**).

---

## ⚡ Performance & Boot Optimizations

- **Zero-Network Import Boot:** NLTK data checks use local disk lookup (`nltk.data.find(...)`) without making blocking network calls.
- **Ultra-Fast Local spaCy Pipeline:** Uses `spacy.blank("en")` with `sentencizer` pipeline for sub-second startup.
- **40x Faster PhraseMatcher Initialization:** Uses `nlp.tokenizer.pipe` to batch tokenize all 5,000+ skills in C (reducing initialization from 20s to <0.5s).
- **Async Background Model Pre-Warming:** `@app.on_event("startup")` pre-loads and warms up PyTorch and Keras models in a background thread so the FastAPI server boots instantly while making API responses take **< 100 milliseconds**.

---

## 📂 Python AI Directory Structure

```
python-ai/
├── README.md                   # AI Microservice Documentation
├── requirements.txt            # Python Dependencies
└── app/
    ├── main.py                 # FastAPI App Server & Routes
    ├── data/
    │   ├── ats_keywords.json   # 500 tracked ATS keywords
    │   ├── skills_db.json      # 5000+ categorized skill vocabulary
    │   └── role_templates/     # Role target skill benchmarks
    ├── ml/
    │   ├── models/             # CVScoringNet, SkillMatcherNet, ATSClassifier, ExperienceDetector
    │   └── weights/            # Pre-trained model weights
    └── services/
        ├── nlp_engine.py       # spaCy + NLTK parsing pipeline
        ├── feature_extractor.py# 35-feature vector builder
        ├── scoring_engine.py   # Master score orchestrator
        └── linkedin_scraper.py # LinkedIn OpenGraph & JSON-LD scraper
```

---

## 🔌 Microservice Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service status, author metadata (`Shubham Kumar Jha`) |
| `GET` | `/health` | AI Engine health check & model readiness status |
| `POST` | `/parse-cv` | Extracts raw text, GitHub handle, and LinkedIn URL from uploaded PDF/DOCX |
| `POST` | `/analyze` | Runs full 5-model AI pipeline and returns explainable score report |
| `GET` | `/github/:username` | Evaluates public GitHub developer profile statistics |

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Server
```bash
uvicorn app.main:app --reload --port 8000
```
Server runs on **http://localhost:8000**. Open **http://localhost:8000/docs** for interactive Swagger documentation.

---

## ✉️ Author & Contact

- **Developer:** Shubham Kumar Jha
- **Email:** [shubhamjha22088@gmail.com](mailto:shubhamjha22088@gmail.com)
- **Credit:** *Made with ❤️ by Shubham Kumar Jha*
