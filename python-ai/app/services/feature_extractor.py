"""
Feature Extractor — Pandas + NumPy
Converts NLP analysis output into the 35-dimensional feature vector
used as input to CVScoringNet (PyTorch FFNN).
"""

import numpy as np
import pandas as pd
from typing import Optional


class FeatureExtractor:
    """
    Converts raw NLP analysis into a structured 35-feature numpy array.
    
    Uses Pandas for intermediate data manipulation and NumPy for
    feature vector construction and normalization.
    """

    def __init__(self):
        self.feature_names = [
            # Contact Info (0-2)
            "has_email", "has_phone", "has_linkedin_url",
            # Sections (3-10)
            "has_summary", "has_experience", "has_education", "has_skills",
            "has_projects", "has_certifications", "has_github_url", "section_count",
            # Content Quality (11-20)
            "word_count_norm", "action_verb_count_norm", "quantified_achievements_norm",
            "bullet_point_count_norm", "avg_sentence_length_norm", "passive_voice_ratio",
            "unique_word_ratio", "readability_score_norm", "technical_term_density",
            "professional_tone_score",
            # Skills (21-25)
            "skill_count_norm", "skill_category_diversity",
            "has_programming_skills", "has_data_skills", "has_cloud_skills",
            # Experience (26-30)
            "years_experience_norm", "job_count_norm", "avg_tenure_norm",
            "has_leadership_keywords", "experience_growth",
            # Education (31-34)
            "education_level_norm", "has_cgpa", "cgpa_norm", "education_relevance",
        ]

    def extract(self, nlp_analysis: dict, raw_text: str = "") -> np.ndarray:
        """
        Convert NLP analysis dict → 35-dim feature vector.
        
        Args:
            nlp_analysis: Output of NLPEngine.full_analysis()
            raw_text: Original CV text for additional feature computation
        Returns:
            features: np.ndarray of shape (35,) dtype=float32
        """
        contact = nlp_analysis.get("contact", {})
        skills_data = nlp_analysis.get("skills", {})
        exp_data = nlp_analysis.get("experience", {})
        quality = nlp_analysis.get("content_quality", {})
        edu = nlp_analysis.get("education", {})
        sections = nlp_analysis.get("sections", {})
        skill_categories = skills_data.get("by_category", {})

        # Use Pandas Series for clean feature construction
        feature_dict = {}

        # ── Contact Info ──
        feature_dict["has_email"] = float(bool(contact.get("email")))
        feature_dict["has_phone"] = float(bool(contact.get("phone")))
        feature_dict["has_linkedin_url"] = float(bool(contact.get("linkedin_url")))

        # ── Sections ──
        feature_dict["has_summary"] = float(sections.get("has_summary", False))
        feature_dict["has_experience"] = float(sections.get("has_experience", False))
        feature_dict["has_education"] = float(sections.get("has_education", False))
        feature_dict["has_skills"] = float(sections.get("has_skills", False))
        feature_dict["has_projects"] = float(sections.get("has_projects", False))
        feature_dict["has_certifications"] = float(sections.get("has_certifications", False))
        feature_dict["has_github_url"] = float(bool(contact.get("github_url")))
        feature_dict["section_count"] = np.clip(sections.get("total_sections", 0) / 7.0, 0, 1)

        # ── Content Quality ──
        word_count = quality.get("word_count", 0)
        feature_dict["word_count_norm"] = np.clip(word_count / 800.0, 0, 1)

        action_verbs = quality.get("action_verb_count", 0)
        feature_dict["action_verb_count_norm"] = np.clip(action_verbs / 20.0, 0, 1)

        quantified = quality.get("quantified_achievements", 0)
        feature_dict["quantified_achievements_norm"] = np.clip(quantified / 15.0, 0, 1)

        # Count bullet points from raw text
        bullet_count = len([l for l in raw_text.split("\n") if l.strip().startswith(("•", "-", "*", "·"))])
        feature_dict["bullet_point_count_norm"] = np.clip(bullet_count / 20.0, 0, 1)

        avg_sent_len = quality.get("avg_words_per_sentence", 15)
        feature_dict["avg_sentence_length_norm"] = np.clip(avg_sent_len / 25.0, 0, 1)

        feature_dict["passive_voice_ratio"] = np.clip(quality.get("passive_voice_ratio", 0), 0, 1)
        feature_dict["unique_word_ratio"] = quality.get("unique_word_ratio", 0.5)

        # Readability: penalize if avg sentence length too long or too short
        readability = 1.0 - abs(avg_sent_len - 15) / 20.0
        feature_dict["readability_score_norm"] = np.clip(readability, 0, 1)

        # Technical term density
        tech_skills_count = len(skill_categories.get("programming_languages", []) +
                                skill_categories.get("frameworks", []) +
                                skill_categories.get("tools", []))
        feature_dict["technical_term_density"] = np.clip(tech_skills_count / 20.0, 0, 1)

        # Professional tone: action verbs / total sentences
        sent_count = quality.get("sentence_count", 1)
        feature_dict["professional_tone_score"] = np.clip(action_verbs / max(sent_count, 1), 0, 1)

        # ── Skills ──
        total_skills = skills_data.get("total_count", 0)
        feature_dict["skill_count_norm"] = np.clip(total_skills / 20.0, 0, 1)
        feature_dict["skill_category_diversity"] = np.clip(
            skills_data.get("category_count", 0) / 8.0, 0, 1
        )

        prog_langs = ["programming_languages", "python", "javascript", "java"]
        feature_dict["has_programming_skills"] = float(
            any(cat in skill_categories for cat in prog_langs)
        )

        data_cats = ["data_science", "machine_learning", "data_analysis", "databases"]
        feature_dict["has_data_skills"] = float(
            any(cat in skill_categories for cat in data_cats)
        )

        cloud_cats = ["cloud", "devops", "aws", "azure", "gcp"]
        feature_dict["has_cloud_skills"] = float(
            any(cat in skill_categories for cat in cloud_cats)
        )

        # ── Experience ──
        years = exp_data.get("total_years", 0)
        feature_dict["years_experience_norm"] = np.clip(years / 10.0, 0, 1)
        job_count = exp_data.get("job_count", 0)
        feature_dict["job_count_norm"] = np.clip(job_count / 5.0, 0, 1)

        months = exp_data.get("total_months", 0)
        avg_tenure = months / max(job_count, 1)
        feature_dict["avg_tenure_norm"] = np.clip(avg_tenure / 36.0, 0, 1)  # 36 months = 3 years

        leadership_kws = ["led", "managed", "supervised", "mentored", "directed", "oversaw"]
        raw_lower = raw_text.lower()
        feature_dict["has_leadership_keywords"] = float(
            any(kw in raw_lower for kw in leadership_kws)
        )

        # Experience growth: multiple roles with increasing seniority signals
        senior_kws = ["senior", "lead", "principal", "head", "director"]
        feature_dict["experience_growth"] = float(
            job_count >= 2 and any(kw in raw_lower for kw in senior_kws)
        )

        # ── Education ──
        feature_dict["education_level_norm"] = edu.get("education_level_norm", 0)
        cgpa = edu.get("cgpa")
        feature_dict["has_cgpa"] = float(cgpa is not None)
        feature_dict["cgpa_norm"] = (cgpa / 10.0) if cgpa else 0.0

        relevance_kws = ["computer science", "information technology", "software", "data science",
                         "artificial intelligence", "machine learning", "electronics"]
        feature_dict["education_relevance"] = float(
            any(kw in raw_lower for kw in relevance_kws)
        )

        # Convert to Pandas Series for validation, then numpy array
        features_series = pd.Series(feature_dict)
        features_array = features_series[self.feature_names].values.astype(np.float32)
        features_array = np.clip(features_array, 0, 1)

        return features_array

    def to_dataframe(self, features: np.ndarray) -> pd.DataFrame:
        """Convert feature array to labeled DataFrame for display/debugging."""
        return pd.DataFrame([features], columns=self.feature_names)

    def get_feature_importance(self, features: np.ndarray) -> pd.DataFrame:
        """
        Returns feature importance ranking (for explainability).
        Higher value = more impact on score.
        """
        weights = np.array([
            # Contact (weight)
            3, 3, 2,
            # Sections
            4, 5, 5, 5, 4, 3, 2, 6,
            # Content
            5, 8, 7, 4, 3, -5, 4, 4, 5, 6,
            # Skills
            8, 5, 6, 5, 4,
            # Experience
            8, 5, 4, 5, 4,
            # Education
            5, 3, 4, 4
        ], dtype=np.float32)

        impact = features * weights
        df = pd.DataFrame({
            "feature": self.feature_names,
            "value": features,
            "impact": impact
        }).sort_values("impact", ascending=False)

        return df


# Singleton
feature_extractor = FeatureExtractor()
