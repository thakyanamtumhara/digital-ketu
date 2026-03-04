"""GitHub auto-persist — saves knowledge files back to the repo after learning.

Railway's filesystem is ephemeral — every deploy wipes local files.
This module uses the GitHub Contents API to commit updated knowledge files
back to the repo, so they survive deploys.

How it works:
1. After apply_knowledge_updates() modifies faq.json/style.json/prompt.json
2. persist_knowledge_files() is called (non-blocking, fire-and-forget)
3. Each modified file is pushed to GitHub via Contents API
4. Next deploy pulls the latest files from Git — learned data preserved!

Required env vars:
- GITHUB_TOKEN: Personal access token with 'repo' scope
- GITHUB_REPO: e.g., "thakyanamtumhara/digital-ketu" (default)
- GITHUB_BRANCH: e.g., "main" (default)
"""

import base64
import json
import logging
import threading
from pathlib import Path

import httpx

from core.config import settings, KNOWLEDGE_DIR, LEARNED_DIR

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Files to persist after knowledge updates
PERSIST_FILES = [
    "knowledge/faq.json",
    "knowledge/style.json",
    "knowledge/prompt.json",
    "knowledge/products.json",
    "knowledge/activity_log.json",
]


def _get_headers() -> dict:
    return {
        "Authorization": f"token {settings.github_token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _persist_file(repo_path: str, local_path: Path, message: str):
    """Push a single file to GitHub using Contents API."""
    if not local_path.exists():
        return

    content = local_path.read_text(encoding="utf-8")
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    url = f"{GITHUB_API}/repos/{settings.github_repo}/contents/{repo_path}"

    # First get the current file's SHA (needed for update)
    try:
        resp = httpx.get(url, headers=_get_headers(), params={"ref": settings.github_branch}, timeout=15)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
        else:
            sha = None  # New file
    except Exception:
        sha = None

    # Push the file
    payload = {
        "message": message,
        "content": encoded,
        "branch": settings.github_branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = httpx.put(url, headers=_get_headers(), json=payload, timeout=30)
        if resp.status_code in (200, 201):
            logger.info(f"[GitPersist] Saved {repo_path} to GitHub")
        else:
            logger.error(f"[GitPersist] Failed {repo_path}: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"[GitPersist] Error saving {repo_path}: {e}")


def persist_knowledge_files(source: str = "auto-learn"):
    """Push all knowledge files to GitHub (non-blocking, runs in background thread).

    Called after apply_knowledge_updates() to save learned data.
    """
    if not settings.github_token:
        logger.debug("[GitPersist] No GITHUB_TOKEN configured — skipping persist")
        return

    def _do_persist():
        message = f"auto: knowledge update from {source}"
        persisted = 0

        for repo_path in PERSIST_FILES:
            local_path = KNOWLEDGE_DIR.parent / repo_path
            _persist_file(repo_path, local_path, message)
            persisted += 1

        # Also persist learned/ files (YouTube extracts, etc.)
        if LEARNED_DIR.exists():
            for f in LEARNED_DIR.iterdir():
                if f.suffix == ".json" and not f.name.startswith("_"):
                    repo_path = f"knowledge/learned/{f.name}"
                    _persist_file(repo_path, f, message)
                    persisted += 1
                elif f.suffix == ".txt":
                    repo_path = f"knowledge/learned/{f.name}"
                    _persist_file(repo_path, f, message)
                    persisted += 1

        # Persist internal tracking files too
        for internal in ["_processed_videos.json", "_backfill_state.json"]:
            internal_path = LEARNED_DIR / internal
            if internal_path.exists():
                _persist_file(f"knowledge/learned/{internal}", internal_path, message)

        logger.info(f"[GitPersist] Done — {persisted} files persisted to GitHub")

    # Run in background thread — don't block the API response
    thread = threading.Thread(target=_do_persist, daemon=True)
    thread.start()


def persist_single_file(repo_path: str, local_path: Path, source: str = "auto"):
    """Push a single file to GitHub (non-blocking).

    Use for individual file updates (e.g., after YouTube learning).
    """
    if not settings.github_token:
        return

    def _do():
        _persist_file(repo_path, local_path, f"auto: {source}")

    thread = threading.Thread(target=_do, daemon=True)
    thread.start()
