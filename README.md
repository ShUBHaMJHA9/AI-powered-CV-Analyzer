# 🤖 SMARRTIF AI — AI-Powered CV & Profile Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https.mit.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![React: 19](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Node: 18+](https://img.shields.io/badge/Node.js-18+-339933?logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)

**SMARRTIF AI** is an enterprise-grade, multi-dimensional AI platform that evaluates candidate Resumes/CVs against target job roles, GitHub developer profiles, and LinkedIn intelligence. Powered by **5 custom Deep Learning & NLP models**, it delivers transparent, explainable scoring breakdown reports in sub-second inference speeds.

> **Made with ❤️ by [Shubham Kumar Jha](mailto:shubhamjha22088@gmail.com)**  
> ✉️ **Contact:** `shubhamjha22088@gmail.com`

---

## 🌟 Key Features

- 📄 **Multi-Format Resume Parsing:** Supports PDF (`PyMuPDF`) & Word DOCX (`python-docx`) text extraction with automatic contact entity recognition (`spaCy NER`).
- 🎯 **Role-Targeted Scoring Engine:** Custom Siamese & FFNN neural networks evaluate skill alignment across 12+ tech roles (Data Scientist, ML Engineer, Full Stack, Cloud Architect, DevOps, etc.).
- 🤖 **ATS Screening Simulation:** Keras Bidirectional LSTM model measures Applicant Tracking System pass probability and identifies top missing keywords.
- 🐙 **Live GitHub Developer Intelligence:** Automatically extracts public repos, total star counts ⭐, followers 👥, forks, and top programming languages distribution.
- 💼 **LinkedIn Profile Intelligence:** Evaluates headline relevance, college standing (`IIT`, `NIT`, `BITS`, `Stanford`, `Degree`), certifications (`AWS`, `PMP`, `TensorFlow`), and profile completeness.
- 📈 **Seniority & Experience Detector:** Keras LSTM + regex engine parses explicit years of experience (`5+ years`), date ranges (`2019-2024`), and title seniority keywords.
- 📊 **Modern Interactive UI Dashboard:** Ultra-premium light-theme dashboard with animated score gauges, 6-axis Radar Charts, language pie charts, and AI priority action recommendations.
- ⚡ **Sub-Second Performance Optimization:** Fast-boot local C pipelines (`spacy.blank("en")`, `nlp.tokenizer.pipe` batching) and async background model pre-warming for instant API responses.

---

## 🏗️ Architecture & Tech Stack

The application is built as a microservice architecture comprising three distinct layers:

```
                  ┌─────────────────────────────────────────┐
                  │       Frontend (React + Vite)           │
                  │  Port 5173 • Recharts • TS • Modern UI  │
                  └────────────────────┬────────────────────┘
                                       │ HTTP / REST
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │     API Gateway (Node.js Express)       │
                  │   Port 5000 • Multer • Axios Proxy      │
                  └────────────────────┬────────────────────┘
                                       │ HTTP / REST
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │   AI Microservice (Python FastAPI)      │
                  │  Port 8000 • PyTorch • Keras • spaCy    │
                  └─────────────────────────────────────────┘
```

| Layer | Technologies | Responsibilities |
|---|---|---|
| **Frontend** | React 19, Vite, TypeScript, Recharts, Lucide/SVG Icons, CSS Tokens | Modern UI dashboard, interactive file upload wizard, step navigation, charts & gauges |
| **Node Backend** | Node.js, Express, Multer, Axios, FormData | API Gateway, file multipart handling, CORS management, request proxying |
| **Python AI Engine** | Python 3.10, FastAPI, PyTorch, Keras/TensorFlow, spaCy, NLTK | 5 DL model scoring inference, spaCy NER, GitHub API client, LinkedIn scraper |

---

## 🧠 Deep Learning & NLP Models Breakdown

The **100-point total score** is computed using 5 specialized ML & DL models:

```
Score (100 Pts Max) = CV Quality (30 Pts) + Skill Match (25 Pts) + Experience (20 Pts) + ATS (15 Pts) + GitHub (10 Pts)
```

| Model Name | Framework | Architecture | Output / Role |
|---|---|---|---|
| **1. CVScoringNet** | PyTorch | 4-Layer Feedforward NN (35 engineered features → BatchNorm → Dropout → Sigmoid) | CV Quality Score (30 Pts Max) |
| **2. SkillMatcherNet** | PyTorch | Siamese BiLSTM Neural Network (Cosine similarity between skills & role requirements) | Skill Alignment & Gap Analysis (25 Pts Max) |
| **3. ATSClassifierModel**| Keras / TF | Bidirectional LSTM (500-keyword presence vector → Dense → Sigmoid) | ATS Pass Probability (15 Pts Max) |
| **4. ExperienceDetector**| Keras / TF | LSTM Multi-class Classifier + Hybrid Regex/Date/Title Parser | Seniority Level [Fresher, Junior, Mid, Senior] (20 Pts Max) |
| **5. GitHub & LinkedIn**  | Custom Python| GitHub REST API & Public Metadata Scraper | Social & Code Intelligence (10 Pts Max each) |

---

## 📂 Project Directory Structure

```
AI-powered CV Analyzer/
├── README.md                   # Project Root Documentation
├── frontend/                   # React 19 + TypeScript + Vite Frontend
│   ├── README.md               # Frontend Documentation
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx             # Main Router App
│       ├── index.css           # Global Theme Tokens & Styles
│       ├── components/         # Navbar, Footer, ScoreGauge
│       └── pages/              # Landing, Analyze Wizard, Dashboard Report
├── node-backend/               # Node.js + Express API Gateway
│   ├── README.md               # Node Backend Documentation
│   ├── package.json
│   ├── uploads/                # Temporary file uploads
│   └── src/
│       └── app.js              # Express Server Router
└── python-ai/                  # Python FastAPI AI Microservice
    ├── README.md               # Python AI Documentation
    ├── requirements.txt
    └── app/
        ├── main.py             # FastAPI App Server & Endpoints
        ├── data/               # Role templates, skills DB, ATS keywords
        ├── ml/
        │   ├── models/         # PyTorch & Keras model implementations
        │   └── weights/        # Trained model weight files
        └── services/           # NLPEngine, ScoringEngine, FeatureExtractor, LinkedInScraper
```

---

## ⚡ Quick Start & Installation

### Prerequisites
- **Node.js**: v18.0.0 or higher
- **Python**: v3.10.0 or higher
- **npm** or **yarn**

### 1. Run Python AI Microservice (Port 8000)
```bash
cd python-ai
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Run Node.js API Gateway (Port 5000)
```bash
cd node-backend
npm install
npm start
```

### 3. Run Frontend Web Application (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser to start using **SMARRTIF AI**!

---

## ✉️ Author & Contact

**SMARRTIF AI** is designed and developed by:

- **Author:** Shubham Kumar Jha
- **Email:** [shubhamjha22088@gmail.com](mailto:shubhamjha22088@gmail.com)
- **Credit:** *Made with ❤️ by Shubham Kumar Jha*

---

## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.
