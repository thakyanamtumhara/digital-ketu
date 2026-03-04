import logging
import os
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# --- Persistent Knowledge Directory ---
# Railway Volume: set KNOWLEDGE_VOLUME_PATH env var to mount path (e.g., /data)
# If not set, falls back to local ./knowledge (dev mode)

_PROJECT_DIR = Path(__file__).parent.parent
_BUNDLED_KNOWLEDGE = _PROJECT_DIR / "knowledge"  # Git-tracked, ships with every deploy
_VOLUME_PATH = os.environ.get("KNOWLEDGE_VOLUME_PATH", "")

if _VOLUME_PATH:
    KNOWLEDGE_DIR = Path(_VOLUME_PATH) / "knowledge"
else:
    KNOWLEDGE_DIR = _BUNDLED_KNOWLEDGE

LEARNED_DIR = KNOWLEDGE_DIR / "learned"


def init_knowledge_volume():
    """Initialize persistent volume with bundled knowledge on first boot.

    On Railway with a volume:
    - First deploy: copies bundled knowledge files to volume
    - Subsequent deploys: volume already has data, only adds NEW files from git
      (doesn't overwrite existing learned data)
    """
    if not _VOLUME_PATH:
        # No volume configured — using local files (dev mode)
        LEARNED_DIR.mkdir(parents=True, exist_ok=True)
        return

    logger.info(f"[Volume] Persistent storage: {KNOWLEDGE_DIR}")

    # Create dirs
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)

    # Copy bundled knowledge files to volume (only if not already there)
    for src_file in _BUNDLED_KNOWLEDGE.glob("*.json"):
        dest_file = KNOWLEDGE_DIR / src_file.name
        if not dest_file.exists():
            shutil.copy2(src_file, dest_file)
            logger.info(f"[Volume] Copied {src_file.name} to volume (first boot)")
        else:
            logger.info(f"[Volume] {src_file.name} already on volume — keeping existing (has learned data)")

    # Copy bundled learned files (if any tracked in git)
    bundled_learned = _BUNDLED_KNOWLEDGE / "learned"
    if bundled_learned.exists():
        for src_file in bundled_learned.iterdir():
            dest_file = LEARNED_DIR / src_file.name
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
                logger.info(f"[Volume] Copied learned/{src_file.name} to volume")

    # Count what's on volume
    knowledge_count = len(list(KNOWLEDGE_DIR.glob("*.json")))
    learned_count = len([f for f in LEARNED_DIR.glob("*.json") if not f.name.startswith("_")])
    logger.info(f"[Volume] Ready: {knowledge_count} knowledge files, {learned_count} learned files")


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

    # Server
    port: int = 8000
    auto_reply_enabled: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
