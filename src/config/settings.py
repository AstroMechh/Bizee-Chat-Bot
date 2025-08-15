import os
from dotenv import load_dotenv

load_dotenv()

class AppSettings:
    """Centralized settings for the Bizee Chatbot project."""
    DATA_DIR = "data"
    RAW_HTML_DIR = os.path.join(DATA_DIR, "raw_html")
    PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")

    STATE_URL_TEMPLATE = os.getenv(
        "BIZEE_STATE_LLC_URL_TEMPLATE",
        "https://bizee.com/{state_slug}-llc"
    )
    STATE_URL_OVERRIDES = {
        # "washington-dc": "https://bizee.com/washington-dc-llc",
    }

    os.makedirs(RAW_HTML_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

def to_state_slug(state: str) -> str:
    return (
        state.lower().strip()
        .replace(" ", "-")
        .replace(".", "")
    )

def state_to_url(state: str) -> str:
    slug = to_state_slug(state)
    overrides = AppSettings.STATE_URL_OVERRIDES
    return overrides.get(slug, AppSettings.STATE_URL_TEMPLATE.format(state_slug=slug))
