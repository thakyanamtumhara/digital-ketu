import logging
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# --- Knowledge Directory ---
_PROJECT_DIR = Path(__file__).parent.parent
KNOWLEDGE_DIR = _PROJECT_DIR / "knowledge"
LEARNED_DIR = KNOWLEDGE_DIR / "learned"


def init_knowledge_dir():
    """Ensure knowledge directories exist on startup."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Knowledge] Dir ready: {KNOWLEDGE_DIR}")


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str = ""

    # WhatsApp Business API
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = "digital-ketu-verify"
    whatsapp_app_secret: str = ""

    # YouTube
    youtube_api_key: str = ""
    youtube_channel_id: str = ""

    # GitHub Auto-Persist (backup — saves learned knowledge back to repo)
    github_token: str = ""
    github_repo: str = "thakyanamtumhara/digital-ketu"
    github_branch: str = "main"

    # PostgreSQL (primary storage — Railway auto-sets DATABASE_URL)
    database_url: str = ""

    # Server
    port: int = 8000
    auto_reply_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
