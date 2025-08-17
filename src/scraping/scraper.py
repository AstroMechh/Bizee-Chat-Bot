import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from src.config.settings import AppSettings, to_state_slug, state_to_url

# Shared session with retry/backoff
_session = requests.Session()
_retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

def fetch_html(url: str, state: str) -> str | None:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = _session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"[fetch_html] {state} {url} -> {e}")
        return None

# Optional DOM hints for odd pages; add entries only if needed.
STATE_SELECTOR_HINTS: dict[str, str] = {
    # "texas": r"main-content|article|page-content",
    # "florida": r"page-body|rich-text|content",
}

def parse_llc_page_generic(html_content: str, selector_regex: str | None = None) -> str:
    soup = BeautifulSoup(html_content, "lxml")

    regex = re.compile(
        selector_regex or r"main-content|article-body|page-content",
        re.IGNORECASE
    )
    main = soup.find("div", class_=regex)
    if not main:
        main = soup.body
    if not main:
        return ""

    for tag in main.find_all(["header","footer","nav","aside","form",
                              "script","style","noscript","img"]):
        tag.decompose()

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def scrape_and_save_state_llc_data(state: str) -> tuple[str | None, str | None]:
    """Scrape one state's Bizee LLC page, save raw HTML and processed text."""
    slug = to_state_slug(state)
    url = state_to_url(state)
    print(f"[scrape] {state} -> {url}")

    html = fetch_html(url, state)
    if not html:
        return None, None

    raw_path = os.path.join(AppSettings.RAW_HTML_DIR, f"{slug}_llc_raw.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  saved raw -> {raw_path}")

    hint = STATE_SELECTOR_HINTS.get(slug)
    processed = parse_llc_page_generic(html, hint)

    text_path = os.path.join(AppSettings.PROCESSED_TEXT_DIR, f"{slug}_llc_content.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(processed)
    print(f"  saved text -> {text_path}  ({len(processed)} chars)")

    return raw_path, text_path


def scrape_states(states: list[str], parallel: bool = False, max_workers: int | None = None):
    """Scrape multiple states, optionally using parallel threads."""
    results: list[tuple[str | None, str | None]] = []
    if parallel:
        workers = max_workers or min(32, len(states)) or 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_state = {ex.submit(scrape_and_save_state_llc_data, s): s for s in states}
            for fut in as_completed(fut_to_state):
                state = fut_to_state[fut]
                try:
                    results.append(fut.result())
                except Exception as e:  # pragma: no cover - defensive
                    print(f"[scrape] {state} failed -> {e}")
                    results.append((None, None))
    else:
        for s in states:
            try:
                results.append(scrape_and_save_state_llc_data(s))
            except Exception as e:  # pragma: no cover - defensive
                print(f"[scrape] {s} failed -> {e}")
                results.append((None, None))
    return results
