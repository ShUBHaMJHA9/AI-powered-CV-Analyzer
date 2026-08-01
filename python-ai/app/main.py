"""
SMARRTIF AI — Python FastAPI Service
Main entry point. Runs on port 8000.
All AI/ML inference happens here using PyTorch + Keras + spaCy + NLTK.
"""

import sys
import os
import json
import re
import io
from pathlib import Path
from typing import Optional

# ── Ensure parent directory is in sys.path ────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

from app.services.linkedin_scraper import linkedin_scraper

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(
    title="SMARRTIF AI — CV Analyzer",
    description="AI-powered CV analysis using PyTorch, Keras, spaCy & NLTK",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"🐍 [PYTHON AI] {request.method} {request.url.path}")
    response = await call_next(request)
    return response

DATA_DIR = BASE_DIR / "data"

# ── Lazy imports for heavy ML engines ──────────────────────────
_nlp_engine = None
_scoring_engine = None


def get_nlp_engine():
    global _nlp_engine
    if _nlp_engine is None:
        from app.services.nlp_engine import NLPEngine
        _nlp_engine = NLPEngine()
    return _nlp_engine


def get_scoring_engine():
    global _scoring_engine
    if _scoring_engine is None:
        from app.services.scoring_engine import ScoringEngine
        _scoring_engine = ScoringEngine()
    return _scoring_engine


@app.on_event("startup")
async def warmup_ai_engines():
    import threading
    def _warmup():
        try:
            get_nlp_engine()
            get_scoring_engine()
            print("[SMARRTIF AI] AI Engines pre-warmed & ready for instant response! 🚀")
        except Exception as e:
            print(f"[SMARRTIF AI Warmup Warning] {e}")
    threading.Thread(target=_warmup, daemon=True).start()


# ── CV Text Extraction ─────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX parsing failed: {e}")


def extract_cv_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")


# ── GitHub Scorer ──────────────────────────────────────────────

def score_github_profile(raw_username: str) -> dict:
    """Score GitHub profile using GitHub public API (proxy-safe)."""
    if not raw_username:
        return {"error": "No GitHub username provided", "weighted_score": 0, "max_points": 10}

    # Clean username from URL or @ syntax (e.g., https://github.com/username/ or @username)
    username = raw_username.strip().rstrip("/")
    if "github.com/" in username.lower():
        username = username.split("github.com/")[-1].split("/")[0].split("?")[0]
    username = username.replace("@", "").strip()

    if not username:
        return {"error": "Invalid GitHub username", "weighted_score": 0, "max_points": 10}

    try:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "SMARRTIF-AI/1.0"}
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        session = requests.Session()
        session.proxies = {"http": None, "https": None}

        # User info request
        r = session.get(f"https://api.github.com/users/{username}", headers=headers, timeout=4)
        if r.status_code == 404:
            return {"error": f"GitHub user '@{username}' not found", "username": username, "weighted_score": 0, "max_points": 10}
        if r.status_code != 200:
            return {"error": "GitHub API rate limit or error", "username": username, "weighted_score": 0, "max_points": 10}

        user = r.json()

        # Repos info request
        repos_r = session.get(
            f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated",
            headers=headers, timeout=5
        )
        repos = repos_r.json() if repos_r.status_code == 200 else []
        if not isinstance(repos, list):
            repos = []

        total_stars = sum(repo.get("stargazers_count", 0) for repo in repos)
        total_forks = sum(repo.get("forks_count", 0) for repo in repos)
        languages = {}
        for repo in repos[:30]:
            lang = repo.get("language")
            if lang:
                languages[lang] = languages.get(lang, 0) + 1

        top_langs = dict(sorted(languages.items(), key=lambda x: x[1], reverse=True)[:6])
        total_lang_count = sum(top_langs.values()) or 1
        lang_pct = {k: round(v / total_lang_count * 100) for k, v in top_langs.items()}

        public_repos = user.get("public_repos", 0)
        followers = user.get("followers", 0)
        bio = user.get("bio") or ""

        # Sub-scores (10 pts total)
        repo_score   = min(public_repos / 15, 1) * 3.0       # 3 pts
        star_score   = min(total_stars / 30, 1) * 2.5        # 2.5 pts
        follow_score = min(followers / 30, 1) * 1.5       # 1.5 pts
        lang_score   = min(len(top_langs) / 4, 1) * 2.0     # 2 pts
        bio_score    = 1.0 if bio else 0.0                    # 1 pt

        total_score = round(min(repo_score + star_score + follow_score + lang_score + bio_score, 10.0), 1)

        return {
            "username": username,
            "name": user.get("name") or username,
            "bio": bio,
            "avatar_url": user.get("avatar_url"),
            "public_repos": public_repos,
            "followers": followers,
            "total_stars": total_stars,
            "total_forks": total_forks,
            "top_languages": lang_pct,
            "weighted_score": total_score,
            "max_points": 10,
            "breakdown": {
                "repos": round(repo_score, 1),
                "stars": round(star_score, 1),
                "followers": round(follow_score, 1),
                "language_diversity": round(lang_score, 1),
                "bio": round(bio_score, 1)
            }
        }
    except Exception as e:
        return {"error": str(e), "weighted_score": 0, "max_points": 10}


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "SMARRTIF AI CV Analyzer",
        "status": "running",
        "version": "1.0.0",
        "author": "Shubham Kumar Jha",
        "credit": "Made with ❤️ by Shubham Kumar Jha",
        "contact_email": "shubhamjha22088@gmail.com"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": "ready",
        "author": "Shubham Kumar Jha",
        "credit": "Made with ❤️ by Shubham Kumar Jha",
        "contact_email": "shubhamjha22088@gmail.com"
    }


@app.post("/parse-cv")
async def parse_cv_quick(cv: UploadFile = File(...)):
    """
    Quick CV parser endpoint: extracts raw text, GitHub URL, and LinkedIn URL.
    Used by frontend to auto-populate GitHub & LinkedIn input fields when user uploads CV!
    """
    file_bytes = await cv.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    raw_text = extract_cv_text(cv.filename or "file.pdf", file_bytes)
    extracted = linkedin_scraper.extract_urls(raw_text)

    # If LinkedIn URL found, scrape public metadata
    linkedin_scraped = {}
    if extracted["linkedin_url"]:
        linkedin_scraped = linkedin_scraper.scrape_profile(extracted["linkedin_url"])

    return JSONResponse(content={
        "raw_text_length": len(raw_text),
        "github_url": extracted["github_url"],
        "github_username": extracted["github_username"],
        "linkedin_url": extracted["linkedin_url"],
        "linkedin_username": extracted["linkedin_username"],
        "linkedin_data": linkedin_scraped
    })


@app.post("/analyze")
async def analyze_cv(
    cv: UploadFile = File(...),
    target_role: str = Form(default="data_scientist"),
    github_username: Optional[str] = Form(default=None),
    linkedin_data: Optional[str] = Form(default=None),
):
    """
    Main CV analysis endpoint.
    1. Extract text from PDF/DOCX
    2. Auto-detect GitHub and LinkedIn URLs if missing
    3. Run NLP pipeline & custom Deep Learning scoring models
    4. Return full explainable score report
    """
    # 1. Read and extract text
    file_bytes = await cv.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    raw_text = extract_cv_text(cv.filename or "file.pdf", file_bytes)
    if len(raw_text) < 30:
        raise HTTPException(status_code=400, detail="Could not extract text from CV.")

    # Auto-extract URLs from resume text
    auto_urls = linkedin_scraper.extract_urls(raw_text)

    # Use auto-detected GitHub username if user didn't specify one
    active_github = (github_username or "").strip() or auto_urls.get("github_username")

    # 2. Fetch GitHub data
    github_result = None
    if active_github:
        github_result = score_github_profile(active_github)

    # 3. LinkedIn data handling & scraping
    linkedin_info = {}
    if linkedin_data:
        try:
            linkedin_info = json.loads(linkedin_data)
        except Exception:
            pass

    # If LinkedIn URL is present, scrape public metadata to enrich profile analysis
    linkedin_url = linkedin_info.get("url") or auto_urls.get("linkedin_url")
    if linkedin_url and not linkedin_info.get("scraped"):
        scraped_linkedin = linkedin_scraper.scrape_profile(linkedin_url)
        linkedin_info.update(scraped_linkedin)

    # 4. Run scoring engine (PyTorch + Keras + spaCy + NLTK)
    engine = get_scoring_engine()
    result = engine.compute_final_score(raw_text, target_role, github_result, linkedin_info)

    # Attach extracted URLs for display
    result["extracted_github_username"] = active_github
    result["extracted_linkedin_url"] = linkedin_url

    return JSONResponse(content=result)


@app.get("/github/{username}")
def github_profile(username: str):
    """Standalone GitHub profile scoring endpoint."""
    return score_github_profile(username)


@app.get("/linkedin")
def linkedin_profile(url: str):
    """Standalone LinkedIn profile scraping endpoint."""
    return linkedin_scraper.scrape_profile(url)


@app.get("/roles")
def list_roles():
    """List all supported target roles."""
    role_dir = DATA_DIR / "role_templates"
    roles = [f.stem for f in role_dir.glob("*.json")]
    return {"roles": sorted(roles)}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
