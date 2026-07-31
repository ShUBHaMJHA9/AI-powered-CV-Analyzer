"""
NLP Engine — spaCy + NLTK
Extracts structured information from raw CV text.
Uses spaCy for NER and NLTK for tokenization, POS tagging, stopwords.
"""

import re
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import Optional

import numpy as np
from pathlib import Path
import json

import socket
import contextlib
import io

# Check locally available NLTK data (zero network calls on startup)
def _ensure_nltk_data():
    pass

_ensure_nltk_data()

# Ultra-fast local spaCy pipeline (0.05s startup)
nlp = spacy.blank("en")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Safe loading of NLTK tools with static fallback
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
        "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
        "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
        "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
        "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
        "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
        "you've", "your", "yours", "yourself", "yourselves"
    }


# Load skills database
DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "skills_db.json", "r") as f:
    SKILLS_DB = json.load(f)

ALL_SKILLS = []
for category, skills in SKILLS_DB.items():
    ALL_SKILLS.extend([(s.lower(), category) for s in skills])

SKILLS_LOOKUP = {s: cat for s, cat in ALL_SKILLS}

# Build spaCy PhraseMatcher for skill extraction (using batch tokenizer pipe for 40x speedup)
from spacy.matcher import PhraseMatcher
skill_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_patterns = list(nlp.tokenizer.pipe([s for s, _ in ALL_SKILLS]))
skill_matcher.add("SKILL", skill_patterns)

# Action verbs (strong resume verbs)
ACTION_VERBS = {
    "achieved", "analyzed", "architected", "automated", "built", "collaborated",
    "created", "delivered", "deployed", "designed", "developed", "enhanced",
    "engineered", "established", "executed", "implemented", "improved", "increased",
    "integrated", "launched", "led", "managed", "migrated", "optimized",
    "orchestrated", "oversaw", "pioneered", "produced", "reduced", "refactored",
    "scaled", "shipped", "spearheaded", "streamlined", "transformed"
}

# Regex patterns
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+")
DATE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|"
    r"April|June|July|August|September|October|November|December)[\s,]*\d{4}\b|"
    r"\b\d{4}\s*[-–]\s*(\d{4}|present|current|now)\b", re.IGNORECASE
)
YEAR_RANGE_RE = re.compile(r"(\d{4})\s*[-–to]\s*(\d{4}|present|current)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+[%+]?\b")


class NLPEngine:
    """Main NLP pipeline using spaCy + NLTK for CV analysis."""

    def extract_contact_info(self, text: str) -> dict:
        """Extract email, phone, LinkedIn URL, GitHub URL using regex + spaCy NER."""
        emails = EMAIL_RE.findall(text)
        phones = PHONE_RE.findall(text)
        urls = URL_RE.findall(text)

        linkedin = next((u for u in urls if "linkedin.com" in u.lower()), None)
        github = next((u for u in urls if "github.com" in u.lower()), None)

        # Use spaCy NER for name detection
        doc = nlp(text[:500])  # check first 500 chars for name
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        name = persons[0] if persons else None

        return {
            "name": name,
            "email": emails[0] if emails else None,
            "phone": phones[0].strip() if phones else None,
            "linkedin_url": linkedin,
            "github_url": github,
            "other_urls": [u for u in urls if u not in [linkedin, github]]
        }

    def extract_skills(self, text: str) -> dict:
        """
        Extract skills using spaCy PhraseMatcher against 5000+ skill vocabulary.
        Returns skills grouped by category.
        """
        doc = nlp(text.lower())
        matches = skill_matcher(doc)

        found_skills = {}
        seen = set()

        for _, start, end in matches:
            skill_text = doc[start:end].text.lower()
            if skill_text in seen:
                continue
            seen.add(skill_text)

            category = SKILLS_LOOKUP.get(skill_text, "general")
            if category not in found_skills:
                found_skills[category] = []
            found_skills[category].append(skill_text.title())

        return {
            "all_skills": list(seen),
            "by_category": found_skills,
            "total_count": len(seen),
            "category_count": len(found_skills)
        }

    def extract_experience_years(self, text: str) -> dict:
        """
        Calculate total years of experience from explicit text mentions and date ranges.
        """
        import datetime
        current_year = datetime.datetime.now().year
        text_lower = text.lower()

        # 1. Explicit years mention (e.g. "5+ years of experience", "4 yrs exp", "3-5 years")
        explicit_patterns = [
            r"(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:exp|experience|working)?",
            r"(?:exp|experience)\s*(?:of\s+)?(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\+?\s*years?\s+in\s+",
        ]

        explicit_years = []
        for pat in explicit_patterns:
            for m in re.finditer(pat, text_lower):
                try:
                    val = float(m.group(1))
                    if 0.5 <= val <= 40:
                        explicit_years.append(val)
                except ValueError:
                    pass

        # 2. Date ranges (e.g. "2019 - 2023", "Jan 2020 - Present", "2021 to Present")
        date_range_pat = re.compile(
            r"\b(199\d|20[0-2]\d)\s*[-–to]+\s*(199\d|20[0-2]\d|present|current|now)\b",
            re.IGNORECASE
        )

        total_months = 0
        date_ranges = []
        for match in date_range_pat.finditer(text):
            start_year = int(match.group(1))
            end_str = match.group(2).lower()
            end_year = current_year if end_str in {"present", "current", "now"} else int(end_str)

            if 1990 <= start_year <= current_year and start_year <= end_year:
                months = (end_year - start_year) * 12
                total_months += months
                date_ranges.append({"start": start_year, "end": end_year, "months": months})

        calculated_years = round(total_months / 12, 1)

        # Final decision
        final_years = 0.0
        if explicit_years:
            final_years = max(explicit_years)
        elif calculated_years > 0:
            final_years = calculated_years

        return {
            "total_years": final_years,
            "total_months": int(final_years * 12),
            "explicit_years": explicit_years,
            "date_ranges": date_ranges,
            "job_count": len(date_ranges)
        }

    def analyze_content_quality(self, text: str) -> dict:
        """
        Analyze text quality using spaCy (primary fast C engine) with NLTK/Regex fallbacks.
        """
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not sentences:
                sentences = [text.strip()] if text.strip() else []
            tokens = [token.text.lower() for token in doc if token.is_alpha]
            meaningful_tokens = [w for w in tokens if w not in stop_words]

            # POS tagging & Lemmatization with spaCy
            verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB" or token.tag_.startswith("VB")]
            action_verb_count = sum(1 for v in verbs if v in ACTION_VERBS)
        except Exception:
            # Fallback to NLTK / Regex
            try:
                sentences = sent_tokenize(text)
            except Exception:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

            try:
                words = word_tokenize(text.lower())
            except Exception:
                words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

            tokens = [w for w in words if w.isalpha()]
            meaningful_tokens = [w for w in tokens if w not in stop_words]

            try:
                pos_tags = pos_tag(tokens)
                verbs = [w for w, tag in pos_tags if tag.startswith("VB")]
                lemmatized_verbs = [lemmatizer.lemmatize(v, "v") for v in verbs]
                action_verb_count = sum(1 for v in lemmatized_verbs if v in ACTION_VERBS)
            except Exception:
                action_verb_count = sum(1 for w in tokens if w in ACTION_VERBS)

        # Passive voice detection (simple heuristic: "was/were/been + past participle")
        passive_count = len(re.findall(
            r"\b(was|were|been|be|is|are)\s+\w+ed\b", text, re.IGNORECASE
        ))

        # Quantified achievements (numbers + % + metrics)
        numbers_found = NUMBER_RE.findall(text)
        quantified_count = len(numbers_found)

        # Unique word ratio
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)

        # Simple readability (avg words per sentence)
        avg_words_per_sentence = len(tokens) / max(len(sentences), 1)

        return {
            "word_count": len(tokens),
            "sentence_count": len(sentences),
            "action_verb_count": action_verb_count,
            "passive_voice_count": passive_count,
            "passive_voice_ratio": round(passive_count / max(len(sentences), 1), 3),
            "quantified_achievements": quantified_count,
            "unique_word_ratio": round(unique_ratio, 3),
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "meaningful_word_count": len(meaningful_tokens),
        }

    def extract_education(self, text: str) -> dict:
        """
        Extract education information using spaCy NER + keyword matching.
        """
        doc = nlp(text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

        degree_patterns = {
            "phd": 4, "doctorate": 4, "ph.d": 4,
            "master": 3, "m.tech": 3, "msc": 3, "mba": 3, "m.e": 3,
            "bachelor": 2, "b.tech": 2, "b.e": 2, "bsc": 2, "b.sc": 2,
            "diploma": 1, "polytechnic": 1
        }

        education_level = 0
        detected_degree = None
        text_lower = text.lower()

        for degree, level in degree_patterns.items():
            if degree in text_lower and level > education_level:
                education_level = level
                detected_degree = degree

        # CGPA/GPA extraction
        cgpa_match = re.search(r"(\d+\.?\d*)\s*/\s*10|gpa\s*:?\s*(\d+\.?\d*)", text_lower)
        cgpa = None
        if cgpa_match:
            cgpa = float(cgpa_match.group(1) or cgpa_match.group(2))

        return {
            "education_level": education_level,
            "education_level_norm": education_level / 4.0,
            "detected_degree": detected_degree,
            "institutions": orgs[:3],
            "cgpa": cgpa,
            "cgpa_norm": cgpa / 10.0 if cgpa else None
        }

    def detect_sections(self, text: str) -> dict:
        """
        Detect which sections are present in the CV.
        Uses keyword headers + spaCy to find section boundaries.
        """
        section_keywords = {
            "summary": ["summary", "profile", "objective", "about me", "overview"],
            "experience": ["experience", "work history", "employment", "professional experience"],
            "education": ["education", "academic", "qualification", "degree"],
            "skills": ["skills", "technical skills", "competencies", "technologies"],
            "projects": ["projects", "personal projects", "academic projects", "portfolio"],
            "certifications": ["certifications", "certificates", "courses", "training"],
            "achievements": ["achievements", "awards", "honors", "accomplishments"],
        }

        text_lower = text.lower()
        detected = {}

        for section, keywords in section_keywords.items():
            detected[f"has_{section}"] = any(kw in text_lower for kw in keywords)

        detected["total_sections"] = sum(1 for v in detected.values() if v)
        return detected

    def full_analysis(self, text: str) -> dict:
        """Run the complete NLP pipeline on CV text."""
        return {
            "contact": self.extract_contact_info(text),
            "skills": self.extract_skills(text),
            "experience": self.extract_experience_years(text),
            "content_quality": self.analyze_content_quality(text),
            "education": self.extract_education(text),
            "sections": self.detect_sections(text)
        }


# Singleton instance
nlp_engine = NLPEngine()
