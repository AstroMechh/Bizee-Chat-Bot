from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.scraping import scraper


def test_collect_tab_urls_basic():
    base_html = (
        "<nav class='tabs'><a href='/overview'>Overview</a>"
        "<a href='fees'>Fees</a></nav>"
    )
    base_url = "https://example.com/state"
    urls = scraper.collect_tab_urls(base_html, base_url)
    assert urls == [
        "https://example.com/overview",
        "https://example.com/fees",
    ]


def test_scrape_and_save_state_llc_data_multi_tab(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    monkeypatch.setattr(scraper.AppSettings, "RAW_HTML_DIR", str(raw_dir))
    monkeypatch.setattr(scraper.AppSettings, "PROCESSED_TEXT_DIR", str(processed_dir))

    base_url = "https://example.com/state"
    base_html = """
    <html><body>
      <nav class="tabs">
        <a href="/overview">Overview</a>
        <a href="fees">Fees</a>
      </nav>
      <div>Main Content</div>
    </body></html>
    """
    tab_htmls = {
        "https://example.com/overview": "<div>Overview Content</div>",
        "https://example.com/fees": "<div>Fees Content</div>",
    }

    def fake_fetch(url: str, state: str) -> str:
        if url == base_url:
            return base_html
        return tab_htmls[url]

    monkeypatch.setattr(scraper, "fetch_html", fake_fetch)
    monkeypatch.setattr(scraper, "state_to_url", lambda s: base_url)

    scraper.scrape_and_save_state_llc_data("Example")

    raw_files = {p.name for p in raw_dir.iterdir()}
    assert raw_files == {
        "example_llc_raw.html",
        "example_overview_raw.html",
        "example_fees_raw.html",
    }

    processed_files = {p.name for p in processed_dir.iterdir()}
    assert processed_files == {
        "example_llc_content.txt",
        "example_overview_content.txt",
        "example_fees_content.txt",
    }

    assert "Overview Content" in (processed_dir / "example_overview_content.txt").read_text()
    assert "Fees Content" in (processed_dir / "example_fees_content.txt").read_text()

