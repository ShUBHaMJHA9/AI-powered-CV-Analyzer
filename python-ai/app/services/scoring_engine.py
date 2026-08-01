"""
Scoring Engine — Master Orchestrator
Combines all 5 DL model outputs into the final transparent score.
Coordinates: CVScoringNet + SkillMatcherNet + ATSClassifier + ExperienceDetector + GitHub
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional

from app.services.nlp_engine import nlp_engine
from app.services.feature_extractor import feature_extractor
from app.ml.models.cv_scorer import CVScoringNet, N_FEATURES
from app.ml.models.skill_matcher import SkillMatcherNet
from app.ml.models.section_clf import SectionClassifierNet, SECTION_LABELS
from app.ml.models.ats_model import ATSClassifier
from app.ml.models.exp_detector import ExperienceDetector

WEIGHTS_DIR = Path(__file__).parent.parent / "ml" / "weights"
DATA_DIR = Path(__file__).parent.parent / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_role_templates() -> dict:
    templates = {}
    role_dir = DATA_DIR / "role_templates"
    for f in role_dir.glob("*.json"):
        templates[f.stem] = json.loads(f.read_text())
    return templates


def _load_ats_keywords() -> list:
    path = DATA_DIR / "ats_keywords.json"
    return json.loads(path.read_text()) if path.exists() else []


class ScoringEngine:
    """
    Master scoring engine that orchestrates all DL models.
    
    Score breakdown (100 pts total):
      CV Quality         30 pts  ← CVScoringNet (PyTorch FFNN)
      Skill Match        25 pts  ← SkillMatcherNet (PyTorch Siamese BiLSTM)
      ATS Compatibility  15 pts  ← ATSClassifierModel (Keras BiLSTM)
      Experience Level   20 pts  ← ExperienceLevelDetector (Keras LSTM)
      GitHub Profile     10 pts  ← Rule-based (PyGithub)
    """

    def __init__(self):
        self.role_templates = _load_role_templates()
        self.ats_keywords = _load_ats_keywords()

        # Load PyTorch models
        self.cv_scorer = self._load_cv_scorer()
        self.skill_matcher = self._load_skill_matcher()

        # Load Keras models
        self.ats_classifier = ATSClassifier(
            ats_keywords=self.ats_keywords,
            weights_path=str(WEIGHTS_DIR / "ats_classifier.keras")
        )
        self.exp_detector = ExperienceDetector(
            weights_path=str(WEIGHTS_DIR / "exp_detector.keras")
        )

        print("[ScoringEngine] All models loaded ✅")

    def _load_cv_scorer(self) -> CVScoringNet:
        model = CVScoringNet(n_features=N_FEATURES).to(DEVICE)
        weights_path = WEIGHTS_DIR / "cv_scorer.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print(f"  [CVScoringNet] Weights loaded from {weights_path}")
        model.eval()
        return model

    def _load_skill_matcher(self) -> SkillMatcherNet:
        vocab_path = WEIGHTS_DIR / "skill_vocab.pt"
        vocab = torch.load(vocab_path) if vocab_path.exists() else {}
        model = SkillMatcherNet(vocab_size=max(len(vocab) + 1, 100)).to(DEVICE)
        weights_path = WEIGHTS_DIR / "skill_matcher.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        self._skill_vocab = vocab
        return model

    def _tokenize_skills(self, text: str, max_len: int = 10) -> torch.Tensor:
        tokens = [self._skill_vocab.get(w.lower(), 0) for w in text.split()]
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return torch.tensor([tokens], dtype=torch.long).to(DEVICE)

    def score_cv_quality(self, nlp_analysis: dict, raw_text: str) -> dict:
        """Score CV quality using CVScoringNet (PyTorch FFNN)."""
        features = feature_extractor.extract(nlp_analysis, raw_text)
        result = self.cv_scorer.predict_with_breakdown(features)
        score_30 = round(result["overall"] * 0.30, 2)  # scale to 30 pts

        return {
            "raw_score": result["overall"],
            "weighted_score": score_30,
            "max_points": 30,
            "breakdown": result["breakdown"],
            "feature_importance": feature_extractor.get_feature_importance(features).head(5).to_dict("records")
        }

    def score_skill_match(self, user_skills: list, target_role: str) -> dict:
        """Score skill alignment using SkillMatcherNet (PyTorch Siamese BiLSTM)."""
        template = self.role_templates.get(target_role, {})
        required_skills = template.get("required_skills", [])
        bonus_skills = template.get("bonus_skills", [])

        user_skill_text = " ".join(user_skills)
        required_text = " ".join(required_skills)
        bonus_text = " ".join(bonus_skills)

        # Encode and compute similarity
        user_enc = self._tokenize_skills(user_skill_text)
        req_enc = self._tokenize_skills(required_text)
        bonus_enc = self._tokenize_skills(bonus_text)

        with torch.no_grad():
            req_similarity = self.skill_matcher(user_enc, req_enc).item()
            bonus_similarity = self.skill_matcher(user_enc, bonus_enc).item()

        # Weighted: required 70%, bonus 30%
        combined = req_similarity * 0.7 + bonus_similarity * 0.3
        score_25 = round(combined * 25, 2)

        # Find present/missing skills
        user_lower = [s.lower() for s in user_skills]
        present = [s for s in required_skills if s.lower() in user_lower]
        missing = [s for s in required_skills if s.lower() not in user_lower]

        return {
            "raw_similarity": round(combined, 4),
            "weighted_score": score_25,
            "max_points": 25,
            "required_similarity": round(req_similarity, 4),
            "bonus_similarity": round(bonus_similarity, 4),
            "skills_present": present,
            "skills_missing": missing[:10],
            "match_percentage": round(combined * 100, 1)
        }

    def score_ats(self, raw_text: str) -> dict:
        """Score ATS compatibility using ATSClassifierModel (Keras BiLSTM)."""
        result = self.ats_classifier.predict(raw_text)
        score_15 = round(result["ats_probability"] * 15, 2)
        result["weighted_score"] = score_15
        result["max_points"] = 15
        return result

    def score_experience(self, exp_text: str, target_role: str) -> dict:
        """Score experience using ExperienceLevelDetector (Keras LSTM) + NLP."""
        detected = self.exp_detector.predict(exp_text)
        template = self.role_templates.get(target_role, {})
        required_level = template.get("min_experience_level", "Junior")

        level_map = {"Fresher": 0, "Junior": 1, "Mid-Level": 2, "Senior": 3}
        detected_idx = detected["level_index"]
        required_idx = level_map.get(required_level, 1)

        # Score based on match: perfect match = full points, under = penalized
        if detected_idx >= required_idx:
            score_factor = 1.0
        else:
            score_factor = max(0, 1.0 - (required_idx - detected_idx) * 0.35)

        score_20 = round(score_factor * 20, 2)

        return {
            "detected_level": detected["level"],
            "required_level": required_level,
            "detected_years": detected.get("detected_years"),
            "confidence": detected["confidence"],
            "weighted_score": score_20,
            "max_points": 20,
            "level_match": detected_idx >= required_idx,
            "probabilities": detected["probabilities"]
        }

    def score_github(self, github_data: Optional[dict]) -> dict:
        """Score GitHub profile (rule-based, 10 pts max)."""
        if not github_data or github_data.get("error"):
            return {
                "weighted_score": 0,
                "max_points": 10,
                "reason": github_data.get("error") if github_data else "No GitHub profile provided",
                "breakdown": {"repos": 0, "stars": 0, "followers": 0, "language_diversity": 0}
            }

        # If already calculated with breakdown by main handler
        if "weighted_score" in github_data and "breakdown" in github_data:
            return {
                "weighted_score": github_data["weighted_score"],
                "max_points": 10,
                "username": github_data.get("username"),
                "name": github_data.get("name"),
                "bio": github_data.get("bio"),
                "avatar_url": github_data.get("avatar_url"),
                "public_repos": github_data.get("public_repos", 0),
                "total_stars": github_data.get("total_stars", 0),
                "followers": github_data.get("followers", 0),
                "top_languages": github_data.get("top_languages", {}),
                "breakdown": github_data["breakdown"]
            }

        repos = github_data.get("public_repos", 0)
        stars = github_data.get("total_stars", 0)
        followers = github_data.get("followers", 0)
        languages = len(github_data.get("top_languages", {}))

        repo_score   = min(repos / 15, 1) * 3.0
        star_score   = min(stars / 30, 1) * 2.5
        follow_score = min(followers / 30, 1) * 1.5
        lang_score   = min(languages / 4, 1) * 3.0

        total = round(min(repo_score + star_score + follow_score + lang_score, 10.0), 1)

        return {
            "weighted_score": total,
            "max_points": 10,
            "username": github_data.get("username"),
            "public_repos": repos,
            "total_stars": stars,
            "followers": followers,
            "top_languages": github_data.get("top_languages", {}),
            "breakdown": {
                "repos": round(repo_score, 1),
                "stars": round(star_score, 1),
                "followers": round(follow_score, 1),
                "language_diversity": round(lang_score, 1)
            }
        }

    def score_linkedin(self, linkedin_data: Optional[dict], target_role: str) -> dict:
        """Score LinkedIn profile across headline, skills, college/education, and certifications (10 pts max)."""
        if not linkedin_data:
            return {
                "weighted_score": 0,
                "max_points": 10,
                "headline": None,
                "has_url": False,
                "reason": "No LinkedIn URL/profile provided",
                "breakdown": {"headline_match": 0, "skills_alignment": 0, "education_college": 0, "certifications": 0}
            }

        url = linkedin_data.get("url") or linkedin_data.get("linkedin_url")
        headline = linkedin_data.get("headline") or linkedin_data.get("title") or ""
        company = linkedin_data.get("company") or ""
        education = linkedin_data.get("education") or linkedin_data.get("college") or ""
        certs = linkedin_data.get("certifications") or []
        skills = linkedin_data.get("skills") or ""
        summary = linkedin_data.get("summary") or ""
        scraped = linkedin_data.get("scraped", False)

        has_url = bool(url and "linkedin.com" in url.lower())

        # 1. Headline & Role Alignment (3.0 pts max)
        role_keywords = target_role.replace("_", " ").split()
        combined_head = f"{headline} {company} {summary}".lower()
        headline_match_count = sum(1 for kw in role_keywords if kw in combined_head)
        
        if headline_match_count > 0:
            headline_score = 3.0
        elif headline and not headline.lower().startswith("professional profile"):
            headline_score = 2.5
        elif has_url:
            headline_score = 2.0
        else:
            headline_score = 1.0

        # 2. Skills Alignment (2.5 pts max)
        if isinstance(skills, str) and len(skills) > 5:
            skills_score = 2.5
        elif isinstance(skills, list) and len(skills) > 0:
            skills_score = 2.5
        elif has_url:
            skills_score = 2.0
        else:
            skills_score = 1.0

        # 3. Education & College Standing (2.5 pts max)
        edu_lower = f"{education} {summary}".lower()
        has_college = any(w in edu_lower for w in ["b.tech", "m.tech", "b.e", "b.s", "m.s", "university", "college", "iit", "nit", "bits", "stanford", "mit", "bachelor", "master", "phd", "degree"])
        if has_college:
            education_score = 2.5
        elif education or has_url:
            education_score = 2.0
        else:
            education_score = 1.0

        # 4. Certifications & Licenses (1.0 pt max)
        if certs or "certified" in summary.lower() or "certificate" in summary.lower():
            cert_score = 1.0
        elif has_url:
            cert_score = 0.8
        else:
            cert_score = 0.0

        # 5. Link & Profile Completeness (1.0 pt max)
        completeness_score = 1.0 if has_url else 0.5

        total = round(min(headline_score + skills_score + education_score + cert_score + completeness_score, 10.0), 1)

        return {
            "weighted_score": total,
            "max_points": 10,
            "url": url,
            "name": linkedin_data.get("name"),
            "headline": headline,
            "company": company,
            "education": education,
            "certifications": certs if isinstance(certs, list) else [certs],
            "skills": skills,
            "has_url": has_url,
            "scraped": scraped,
            "breakdown": {
                "headline_match": round(headline_score, 1),
                "skills_alignment": round(skills_score, 1),
                "education_college": round(education_score, 1),
                "certifications": round(cert_score, 1)
            }
        }

    def compute_final_score(
        self,
        raw_text: str,
        target_role: str,
        github_data: Optional[dict] = None,
        linkedin_data: Optional[dict] = None
    ) -> dict:
        """
        Master scoring function — runs all scoring dimensions.
        Returns complete, explainable score report.
        """
        # 1. NLP Analysis
        nlp_analysis = nlp_engine.full_analysis(raw_text)
        user_skills = nlp_analysis["skills"]["all_skills"]
        exp_text = raw_text

        # 2. Run all scoring dimensions
        cv_result = self.score_cv_quality(nlp_analysis, raw_text)
        skill_result = self.score_skill_match(user_skills, target_role)
        ats_result = self.score_ats(raw_text)
        exp_result = self.score_experience(exp_text, target_role)
        github_result = self.score_github(github_data)
        linkedin_result = self.score_linkedin(linkedin_data, target_role)

        # 3. Total score (scaled out of 100)
        raw_total = (
            cv_result["weighted_score"] +
            skill_result["weighted_score"] +
            ats_result["weighted_score"] +
            exp_result["weighted_score"] +
            github_result["weighted_score"]
        )
        total = round(min(raw_total, 100.0), 1)

        # 4. Grade
        if total >= 85:   grade = "A+"
        elif total >= 75: grade = "A"
        elif total >= 65: grade = "B+"
        elif total >= 55: grade = "B"
        elif total >= 45: grade = "C"
        else:             grade = "D"

        return {
            "overall_score": total,
            "grade": grade,
            "target_role": target_role,
            "dimensions": {
                "cv_quality": cv_result,
                "skill_match": skill_result,
                "ats_compatibility": ats_result,
                "experience_level": exp_result,
                "github_profile": github_result,
                "linkedin_profile": linkedin_result
            },
            "nlp_analysis": nlp_analysis,
            "github_data": github_data if (github_data and not github_data.get("error")) else None,
            "github_languages": github_data.get("top_languages", {}) if github_data else {},
            "linkedin_data": linkedin_result,
            "skill_gaps": skill_result["skills_missing"],
            "top_recommendations": self._generate_recommendations(
                total, skill_result, ats_result, cv_result, exp_result
            )
        }

    def _generate_recommendations(self, total, skill_result, ats_result, cv_result, exp_result) -> list:
        """Generate prioritized improvement recommendations."""
        recs = []

        if skill_result["weighted_score"] < 15:
            missing = skill_result["skills_missing"][:3]
            recs.append({
                "priority": 1, "type": "skills",
                "action": f"Add missing skills: {', '.join(missing)}",
                "impact": "High"
            })
        if ats_result["ats_score"] < 60:
            missing_kws = ats_result.get("keywords_missing", [])[:3]
            recs.append({
                "priority": 2, "type": "ats",
                "action": f"Include ATS keywords: {', '.join(missing_kws)}",
                "impact": "High"
            })
        if cv_result["weighted_score"] < 18:
            recs.append({
                "priority": 3, "type": "cv_quality",
                "action": "Add quantified achievements (numbers, %, metrics) to experience bullets",
                "impact": "Medium"
            })
        if not exp_result["level_match"]:
            recs.append({
                "priority": 4, "type": "experience",
                "action": f"This role requires {exp_result['required_level']} experience. Highlight relevant projects.",
                "impact": "Medium"
            })

        return sorted(recs, key=lambda x: x["priority"])


# Singleton (loaded at app startup)
scoring_engine = ScoringEngine()
