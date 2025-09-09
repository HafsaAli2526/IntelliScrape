import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def scrape_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

    # Extract visible text (you can improve this logic later)
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)

        return text.strip() or "No readable text found on page."

    except Exception as e:
        return f"❌ Error while scraping: {str(e)}"
