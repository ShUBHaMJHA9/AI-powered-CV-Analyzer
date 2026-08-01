"""
LinkedIn Scraper & Profile Intelligence Module
Extracts public profile information from LinkedIn URLs and CV text.
Uses crawler User-Agents + BeautifulSoup with fallback to extract profile metadata.
"""

import re
import os
import json
import random
from typing import Optional, Dict
import requests
from bs4 import BeautifulSoup

# Regex patterns for LinkedIn & GitHub URLs
LINKEDIN_URL_RE = re.compile(
    r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE
)
GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE
)

# Search engine crawler User-Agents (bypasses LinkedIn HTTP 999 anti-scraping block)
CRAWLER_USER_AGENTS = [
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)",
    "LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)",
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


class LinkedInScraper:
    """Scrapes public LinkedIn profile data and extracts URLs from text."""

    @staticmethod
    def extract_urls(text: str) -> Dict[str, Optional[str]]:
        """
        Extract LinkedIn and GitHub URLs and usernames from raw text (e.g. CV).
        """
        linkedin_match = LINKEDIN_URL_RE.search(text)
        github_match = GITHUB_URL_RE.search(text)

        linkedin_url = linkedin_match.group(0) if linkedin_match else None
        linkedin_user = linkedin_match.group(1) if linkedin_match else None

        github_url = github_match.group(0) if github_match else None
        github_user = github_match.group(1) if github_match else None

        return {
            "linkedin_url": linkedin_url,
            "linkedin_username": linkedin_user,
            "github_url": github_url,
            "github_username": github_user,
        }

    def scrape_profile(self, linkedin_url: str, cv_text: str = "") -> Dict:
        """
        Scrape public LinkedIn profile metadata via OpenGraph, HTML tags & JSON-LD.
        Safe against network errors and anti-scraping blocks with intelligent fallback.
        """
        if not linkedin_url:
            return self._fallback_data("unknown", "No LinkedIn URL provided", cv_text)

        # Ensure full URL format
        if not linkedin_url.startswith("http"):
            linkedin_url = f"https://www.linkedin.com/in/{linkedin_url.strip('/')}/"

        username_match = LINKEDIN_URL_RE.search(linkedin_url)
        username = username_match.group(1) if username_match else "unknown"

        # Clean username numbers if present (e.g. shubhamjha2005 -> Shubham Jha)
        clean_name_from_user = re.sub(r"\d+", "", username).replace("-", " ").strip().title()
        if not clean_name_from_user:
            clean_name_from_user = username.replace("-", " ").title()

        session = requests.Session()
        session.proxies = {"http": None, "https": None}

        # Try crawler User-Agents sequentially
        for ua in CRAWLER_USER_AGENTS:
            headers = {
                "User-Agent": ua,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            try:
                resp = session.get(linkedin_url, headers=headers, timeout=5)
                if resp.status_code == 200 and len(resp.text) > 1000:
                    soup = BeautifulSoup(resp.text, "html.parser")

                    # Extract OpenGraph & HTML Meta Tags
                    og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
                    og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                    og_image = soup.find("meta", property="og:image")

                    page_title = soup.title.string if soup.title else ""
                    title_text = (og_title["content"] if og_title and "content" in og_title.attrs else page_title).strip()
                    desc_text = (og_desc["content"] if og_desc and "content" in og_desc.attrs else "").strip()
                    image_url = og_image["content"] if og_image and "content" in og_image.attrs else None

                    # Parse title: "Name - Headline | LinkedIn" or "Name | LinkedIn"
                    name = clean_name_from_user
                    headline = ""
                    if " - " in title_text:
                        parts = title_text.split(" - ")
                        name = parts[0].strip()
                        headline = parts[1].split(" | ")[0].strip()
                    elif " | " in title_text:
                        headline = title_text.split(" | ")[0].strip()
                    elif title_text:
                        headline = title_text

                    # Clean headline from title duplicates
                    if headline.lower().startswith(name.lower()):
                        headline = headline[len(name):].strip(" -|:")

                    # Look for JSON-LD structured data
                    json_ld_data = {}
                    education_info = None
                    certifications_info = []

                    for script in soup.find_all("script", type="application/ld+json"):
                        try:
                            data = json.loads(script.string or "{}")
                            if isinstance(data, dict) and data.get("@type") == "Person":
                                json_ld_data = data
                                alumni = data.get("alumniOf")
                                if isinstance(alumni, list) and alumni:
                                    education_info = ", ".join([a.get("name", "") for a in alumni if isinstance(a, dict)])
                                elif isinstance(alumni, dict):
                                    education_info = alumni.get("name")
                                break
                        except Exception:
                            pass

                    job_title = json_ld_data.get("jobTitle") or headline
                    works_for = json_ld_data.get("worksFor", {})
                    company = works_for.get("name") if isinstance(works_for, dict) else None

                    # Additional HTML Element Extraction
                    if not headline:
                        h1 = soup.find("h1")
                        if h1:
                            name = h1.get_text(strip=True)
                        h2 = soup.find("h2")
                        if h2:
                            headline = h2.get_text(strip=True)

                    # Extract education from description
                    if not education_info and desc_text:
                        edu_match = re.search(r"(?:education|studied at|degree from|alumni of|university|college|iit|nit|bits|stanford|mit)\s*:?\s*([^.\n|]+)", desc_text, re.IGNORECASE)
                        if edu_match:
                            education_info = edu_match.group(0).strip()

                    # Extract certifications
                    if desc_text:
                        cert_matches = re.findall(r"\b(AWS Certified\w*|PMP|Scrum Master|TensorFlow Certified\w*|Azure\w*|GCP\w*|Oracle Certified\w*|Certified\s+[A-Za-z0-9\s]+)\b", desc_text, re.IGNORECASE)
                        certifications_info = list(set(cert_matches))

                    # Ensure we don't return blank or generic headline
                    if not headline or headline.lower() in ["linkedin", "profile", "professional profile"]:
                        headline = self._extract_headline_from_cv(cv_text) or f"Software Professional ({name})"

                    return {
                        "scraped": True,
                        "username": username,
                        "linkedin_url": linkedin_url,
                        "name": name or clean_name_from_user,
                        "headline": headline,
                        "company": company,
                        "education": education_info,
                        "certifications": certifications_info,
                        "summary": desc_text,
                        "image_url": image_url,
                        "source": "public_meta_scraper"
                    }
            except Exception:
                continue

        return self._fallback_data(username, "Protected Profile", cv_text)

    def _extract_headline_from_cv(self, cv_text: str) -> Optional[str]:
        """Extract candidate headline/title directly from CV text if public scraping is blocked."""
        if not cv_text:
            return None
        lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
        for line in lines[:12]:
            if re.search(r"\b(Software Engineer|Developer|Data Scientist|Full Stack|Frontend|Backend|ML Engineer|DevOps|Student|Analyst|Intern|Architect|Engineer)\b", line, re.IGNORECASE):
                if len(line) < 90 and not line.startswith("http"):
                    return line
        return None

    def _fallback_data(self, username: str, reason: str, cv_text: str = "") -> Dict:
        """Fallback response when scraping fails or is blocked."""
        clean_name = re.sub(r"\d+", "", username).replace("-", " ").strip().title()
        if not clean_name:
            clean_name = username.replace("-", " ").title()

        extracted_headline = self._extract_headline_from_cv(cv_text)
        headline = extracted_headline if extracted_headline else f"Software Professional ({clean_name})"

        return {
            "scraped": False,
            "username": username,
            "linkedin_url": f"https://www.linkedin.com/in/{username}/",
            "name": clean_name,
            "headline": headline,
            "reason": reason,
            "source": "fallback_enrichment"
        }


# Singleton instance
linkedin_scraper = LinkedInScraper()
