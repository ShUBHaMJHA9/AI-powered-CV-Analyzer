"""
LinkedIn Scraper & Profile Intelligence Module
Extracts public profile information from LinkedIn URLs and CV text.
Uses requests + BeautifulSoup with proxy safety to parse public metadata & JSON-LD.
"""

import re
import os
import json
from typing import Optional, Dict
import requests
from bs4 import BeautifulSoup

# Regex patterns for LinkedIn URLs
LINKEDIN_URL_RE = re.compile(
    r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE
)
GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE
)

# Standard User-Agent header
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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

    def scrape_profile(self, linkedin_url: str) -> Dict:
        """
        Scrape public LinkedIn profile metadata via OpenGraph & JSON-LD tags.
        Safe against network errors and anti-scraping blocks with fallback.
        """
        if not linkedin_url:
            return {"scraped": False, "reason": "No LinkedIn URL provided"}

        # Ensure full URL format
        if not linkedin_url.startswith("http"):
            linkedin_url = f"https://www.linkedin.com/in/{linkedin_url.strip('/')}/"

        username_match = LINKEDIN_URL_RE.search(linkedin_url)
        username = username_match.group(1) if username_match else "unknown"

        # Prepare request session without local broken proxy
        session = requests.Session()
        session.proxies = {"http": None, "https": None}

        try:
            resp = session.get(linkedin_url, headers=HEADERS, timeout=2)
            if resp.status_code != 200:
                return self._fallback_data(username, f"HTTP Status {resp.status_code}")

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract OpenGraph meta tags
            og_title = soup.find("meta", property="og:title")
            og_desc = soup.find("meta", property="og:description")
            og_image = soup.find("meta", property="og:image")

            title_text = og_title["content"] if og_title and "content" in og_title.attrs else ""
            desc_text = og_desc["content"] if og_desc and "content" in og_desc.attrs else ""
            image_url = og_image["content"] if og_image and "content" in og_image.attrs else None

            # Parse title: "Name - Headline | LinkedIn"
            name = username.replace("-", " ").title()
            headline = ""
            if " - " in title_text:
                parts = title_text.split(" - ")
                name = parts[0].strip()
                headline = parts[1].split(" | ")[0].strip()
            elif title_text:
                headline = title_text.split(" | ")[0].strip()

            # Look for JSON-LD structured data
            json_ld_data = {}
            education_info = None
            certifications_info = []
            skills_info = []

            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "{}")
                    if data.get("@type") == "Person":
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

            # Extract college/education from description if not in JSON-LD
            if not education_info and desc_text:
                edu_match = re.search(r"(?:education|studied at|degree from|alumni of|university|college|iit|nit|bits|stanford|mit)\s*:?\s*([^.\n|]+)", desc_text, re.IGNORECASE)
                if edu_match:
                    education_info = edu_match.group(0).strip()

            # Extract certification mentions
            if desc_text:
                cert_matches = re.findall(r"\b(AWS Certified\w*|PMP|Scrum Master|TensorFlow Certified\w*|Azure\w*|GCP\w*|Oracle Certified\w*|Certified\s+[A-Za-z0-9\s]+)\b", desc_text, re.IGNORECASE)
                certifications_info = list(set(cert_matches))

            return {
                "scraped": True,
                "username": username,
                "linkedin_url": linkedin_url,
                "name": name,
                "headline": headline or job_title,
                "company": company,
                "education": education_info,
                "certifications": certifications_info,
                "summary": desc_text,
                "image_url": image_url,
                "source": "public_meta_scraper"
            }

        except Exception as e:
            return self._fallback_data(username, str(e))

    def _fallback_data(self, username: str, reason: str) -> Dict:
        """Fallback response when scraping fails or is blocked."""
        name = username.replace("-", " ").title()
        return {
            "scraped": False,
            "username": username,
            "linkedin_url": f"https://www.linkedin.com/in/{username}/",
            "name": name,
            "headline": f"Professional Profile ({name})",
            "reason": reason,
            "source": "fallback"
        }


# Singleton instance
linkedin_scraper = LinkedInScraper()
