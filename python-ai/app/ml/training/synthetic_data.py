"""
Synthetic Training Data Generator
Generates training data for all 5 DL models without needing a labeled dataset.
Uses rule-based heuristics to create realistic CV feature vectors and labels.
"""

import numpy as np
import json
import random
import re
from pathlib import Path
from typing import Tuple, List

random.seed(42)
np.random.seed(42)


# ─── Section Classifier Training Data ────────────────────────────────────────

SECTION_TEMPLATES = {
    "contact": [
        "John Smith | john.smith@gmail.com | +91 9876543210 | linkedin.com/in/johnsmith",
        "Email: priya.sharma@outlook.com | Phone: 9988776655 | GitHub: github.com/priya",
        "Rahul Kumar\nrahul.kumar@yahoo.com\n+91-8765432109",
        "Name: Alice Johnson | alice@email.com | Mumbai, India",
        "Contact: david@gmail.com | Mobile: +91 7654321098",
    ],
    "summary": [
        "Results-driven data scientist with 3+ years of experience in machine learning and NLP.",
        "Passionate software engineer with expertise in Python, Java, and cloud technologies.",
        "Dynamic professional with strong analytical skills and proven track record in data analysis.",
        "Motivated fresher with hands-on experience in web development and machine learning projects.",
        "Experienced backend developer specializing in scalable microservices and REST APIs.",
    ],
    "experience": [
        "Software Engineer at TCS (2021-2023): Developed REST APIs using Django and PostgreSQL.",
        "Data Analyst | Infosys | Jan 2022 – Present: Built dashboards using Power BI and Tableau.",
        "Machine Learning Engineer, Wipro: Implemented NLP models using BERT and PyTorch.",
        "Full Stack Developer at Startup XYZ: Led frontend team using React.js and Node.js.",
        "Internship - Data Science: Worked on customer churn prediction using scikit-learn.",
    ],
    "education": [
        "B.Tech Computer Science, IIT Delhi, 2023 | CGPA: 8.9/10",
        "Master of Science in Data Science, BITS Pilani, 2022 | GPA: 3.8/4.0",
        "Bachelor of Engineering (Information Technology), Mumbai University, 2021",
        "B.Sc Mathematics (Honors), Delhi University, 2020 | 75%",
        "MBA (Marketing & Analytics), IIM Ahmedabad, 2023",
    ],
    "skills": [
        "Python, Machine Learning, TensorFlow, Keras, PyTorch, Scikit-learn, Pandas, NumPy",
        "JavaScript, React.js, Node.js, Express.js, MongoDB, PostgreSQL, REST APIs",
        "Data Analysis, SQL, Power BI, Tableau, Excel, R, Statistics",
        "Java, Spring Boot, Microservices, Docker, Kubernetes, AWS, CI/CD",
        "NLP, BERT, SpaCy, NLTK, Text Classification, Named Entity Recognition",
    ],
    "projects": [
        "Sentiment Analysis Tool: Built a BERT-based sentiment classifier achieving 93% accuracy.",
        "E-Commerce Platform: Developed full-stack app using MERN stack with 500+ daily users.",
        "Stock Price Predictor: LSTM model for time series forecasting with 85% directional accuracy.",
        "Resume Parser: NLP pipeline using SpaCy to extract structured data from CVs.",
        "Real-time Chat App: WebSocket-based chat application using Node.js and Socket.io.",
    ],
    "certifications": [
        "AWS Certified Solutions Architect – Associate (2023)",
        "Google Professional Data Engineer Certificate | Coursera",
        "TensorFlow Developer Certificate – Google (2022)",
        "Microsoft Azure Fundamentals (AZ-900) | 2023",
        "Deep Learning Specialization – Coursera (Andrew Ng) | 2022",
    ],
    "other": [
        "Volunteer at NGO for digital literacy | Hobby: Open source contributions",
        "Languages: English (Fluent), Hindi (Native), French (Beginner)",
        "References available upon request.",
        "Interests: Machine Learning research, competitive programming, hiking",
        "Activities: Hackathon winner (Smart India Hackathon 2022), IEEE member",
    ]
}


def generate_section_data(n_per_class: int = 500) -> Tuple[List[str], List[int]]:
    """Generate synthetic text samples for section classification training."""
    from ml.models.section_clf import SECTION_LABELS

    texts, labels = [], []
    augmentations = [
        lambda t: t.upper(),
        lambda t: t.lower(),
        lambda t: t + " " + random.choice([".", "-", "|"]) + " Additional info",
        lambda t: t,  # original
        lambda t: t,  # original (duplicate for balance)
    ]

    for label_idx, label in enumerate(SECTION_LABELS):
        templates = SECTION_TEMPLATES.get(label, SECTION_TEMPLATES["other"])
        for _ in range(n_per_class):
            base = random.choice(templates)
            aug = random.choice(augmentations)
            texts.append(aug(base))
            labels.append(label_idx)

    return texts, labels


# ─── CV Scorer Training Data ──────────────────────────────────────────────────

def generate_cv_features(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CV feature vectors with quality scores.
    Uses rule-based scoring to label each feature vector.
    """
    from ml.models.cv_scorer import N_FEATURES

    features_list = []
    scores_list = []

    for _ in range(n_samples):
        f = np.zeros(N_FEATURES, dtype=np.float32)

        # Contact info (indices 0-2)
        f[0] = random.choices([0, 1], weights=[0.1, 0.9])[0]   # has_email
        f[1] = random.choices([0, 1], weights=[0.2, 0.8])[0]   # has_phone
        f[2] = random.choices([0, 1], weights=[0.5, 0.5])[0]   # has_linkedin

        # Sections (3-10)
        f[3] = random.choices([0, 1], weights=[0.4, 0.6])[0]   # has_summary
        f[4] = random.choices([0, 1], weights=[0.1, 0.9])[0]   # has_experience
        f[5] = random.choices([0, 1], weights=[0.05, 0.95])[0] # has_education
        f[6] = random.choices([0, 1], weights=[0.1, 0.9])[0]   # has_skills
        f[7] = random.choices([0, 1], weights=[0.4, 0.6])[0]   # has_projects
        f[8] = random.choices([0, 1], weights=[0.5, 0.5])[0]   # has_certifications
        f[9] = random.choices([0, 1], weights=[0.5, 0.5])[0]   # has_github
        f[10] = np.sum(f[3:10]) / 7.0                           # section_count_norm

        # Content quality (11-20)
        f[11] = np.clip(np.random.normal(0.6, 0.3), 0, 1)      # word_count_norm
        f[12] = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # action_verb_count
        f[13] = np.clip(np.random.normal(0.4, 0.3), 0, 1)      # quantified_achievements
        f[14] = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # bullet_count
        f[15] = np.clip(np.random.normal(0.6, 0.2), 0, 1)      # avg_sentence_length
        f[16] = np.clip(np.random.normal(0.2, 0.15), 0, 1)     # passive_voice (lower better)
        f[17] = np.clip(np.random.normal(0.6, 0.2), 0, 1)      # unique_word_ratio
        f[18] = np.clip(np.random.normal(0.6, 0.2), 0, 1)      # readability
        f[19] = np.clip(np.random.normal(0.4, 0.2), 0, 1)      # tech_term_density
        f[20] = np.clip(np.random.normal(0.6, 0.2), 0, 1)      # professional_tone

        # Skills (21-25)
        f[21] = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # skill_count
        f[22] = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # skill_diversity
        f[23] = random.choices([0, 1], weights=[0.2, 0.8])[0]  # has_programming
        f[24] = random.choices([0, 1], weights=[0.4, 0.6])[0]  # has_data_skills
        f[25] = random.choices([0, 1], weights=[0.5, 0.5])[0]  # has_cloud

        # Experience (26-30)
        f[26] = np.clip(np.random.exponential(0.3), 0, 1)      # years_exp
        f[27] = np.clip(np.random.normal(0.4, 0.3), 0, 1)      # job_count
        f[28] = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # avg_tenure
        f[29] = random.choices([0, 1], weights=[0.6, 0.4])[0]  # leadership
        f[30] = random.choices([0, 1], weights=[0.7, 0.3])[0]  # growth

        # Education (31-34)
        f[31] = random.choice([0, 0.25, 0.5, 0.75, 1.0])       # education_level
        f[32] = random.choices([0, 1], weights=[0.3, 0.7])[0]  # has_cgpa
        f[33] = np.clip(np.random.normal(0.8, 0.15), 0, 1)     # cgpa_norm
        f[34] = random.choices([0, 1], weights=[0.3, 0.7])[0]  # education_relevance

        # Rule-based score calculation (used as training label)
        score = (
            (f[0] + f[1] + f[2]) / 3 * 10 +               # contact: 10 pts
            f[10] * 12 +                                    # sections: 12 pts
            (f[11] + f[12] + f[13] + f[14]) / 4 * 30 +    # content: 30 pts
            (f[21] + f[22] + f[23]) / 3 * 25 +            # skills: 25 pts
            (f[26] + f[27]) / 2 * 20 +                    # experience: 20 pts
            f[31] * 10 - f[16] * 5                         # edu bonus, passive penalty
        )
        score = np.clip(score / 100, 0, 1)

        features_list.append(f)
        scores_list.append([score])

    return np.array(features_list), np.array(scores_list)


# ─── ATS Training Data ────────────────────────────────────────────────────────

def generate_ats_data(ats_keywords: list, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic keyword presence vectors with ATS pass/fail labels."""
    n_keywords = len(ats_keywords)
    X, y = [], []

    for _ in range(n_samples):
        # Random keyword presence ratio
        match_ratio = np.random.beta(2, 3)  # biased toward lower match rates
        vector = (np.random.random(n_keywords) < match_ratio).astype(np.float32)

        # ATS pass if > 40% keyword coverage
        label = 1.0 if match_ratio >= 0.4 else 0.0

        X.append(vector)
        y.append([label])

    return np.array(X), np.array(y)


# ─── Experience Level Training Data ──────────────────────────────────────────

def generate_experience_texts() -> Tuple[List[str], List[int]]:
    """Generate synthetic experience texts for each seniority level."""
    data = {
        0: [  # Fresher
            "Final year B.Tech student. No prior work experience. Completed internship in web dev.",
            "Recent graduate with academic projects in Python and machine learning.",
            "Fresher with 0 years of industry experience. Strong academic background.",
            "Just completed my degree. Looking for entry level position in data science.",
        ],
        1: [  # Junior (0.5-2 years)
            "1 year experience as junior software engineer at startup. Built REST APIs.",
            "1.5 years working as data analyst. Proficient in SQL and Power BI.",
            "Software developer with 2 years experience in React and Node.js.",
            "2 years experience in machine learning model development at MNC.",
        ],
        2: [  # Mid-Level (2-5 years)
            "3 years experience as full stack developer. Led team of 3 developers.",
            "4 years in data science with expertise in NLP and computer vision projects.",
            "Software engineer with 5 years experience in Java Spring Boot microservices.",
            "3.5 years as ML engineer building and deploying production ML systems.",
        ],
        3: [  # Senior (5+ years)
            "Senior software architect with 8 years experience designing scalable systems.",
            "7 years in data engineering and big data platforms (Spark, Kafka, Airflow).",
            "10+ years of experience leading cross-functional engineering teams.",
            "Senior data scientist with 6 years. Published 3 research papers in NLP.",
        ]
    }

    texts, labels = [], []
    for level, examples in data.items():
        # Augment by shuffling and repeating
        augmented = examples * 100 + [e + " " + random.choice(examples) for e in examples] * 50
        for text in augmented:
            texts.append(text)
            labels.append(level)

    return texts, labels
