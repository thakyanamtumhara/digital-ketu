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


def restore_knowledge_from_github():
    """Restore evolved knowledge from GitHub main branch on startup.

    Railway deploys are ephemeral — learned data (evolved traits, patterns,
    learned files) might not be in the deployed branch. This function
    fetches the latest versions from main (where git_persist saves them)
    and merges evolved data into local files.

    Smart merge: only takes evolved/learned parts, doesn't overwrite structure.
    """
    if not settings.github_token:
        logger.debug("[GitRestore] No GITHUB_TOKEN — skipping restore")
        return

    logger.info("[GitRestore] Checking GitHub main branch for evolved knowledge...")

    restored = []

    # 1. Restore prompt.json — merge evolved traits/phrases/rules
    try:
        remote_prompt = _fetch_github_file("knowledge/prompt.json")
        if remote_prompt:
            remote_data = json.loads(remote_prompt)
            local_path = KNOWLEDGE_DIR / "prompt.json"
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)

            remote_version = remote_data.get("version", 1)
            local_version = local_data.get("version", 1)

            if remote_version > local_version:
                # Merge evolved data from remote into local
                for key in ["evolved_traits", "evolved_phrases", "evolved_rules", "evolution_log"]:
                    remote_val = remote_data.get(key, [])
                    if remote_val:
                        local_data[key] = remote_val

                local_data["version"] = remote_version

                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(local_data, f, indent=2, ensure_ascii=False)

                restored.append(f"prompt.json v{local_version} → v{remote_version}")
    except Exception as e:
        logger.error(f"[GitRestore] prompt.json restore error: {e}")

    # 2. Restore style.json — merge learned_patterns + example_conversations
    try:
        remote_style = _fetch_github_file("knowledge/style.json")
        if remote_style:
            remote_data = json.loads(remote_style)
            local_path = KNOWLEDGE_DIR / "style.json"
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)

            # Merge learned_patterns
            remote_patterns = remote_data.get("learned_patterns", [])
            local_patterns = local_data.get("learned_patterns", [])
            if len(remote_patterns) > len(local_patterns):
                local_data["learned_patterns"] = remote_patterns
                restored.append(f"style patterns: {len(local_patterns)} → {len(remote_patterns)}")

            # Merge example_conversations (keep the larger set)
            remote_examples = remote_data.get("example_conversations", [])
            local_examples = local_data.get("example_conversations", [])
            if len(remote_examples) > len(local_examples):
                local_data["example_conversations"] = remote_examples
                restored.append(f"examples: {len(local_examples)} → {len(remote_examples)}")

            if any("style" in r for r in restored):
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(local_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[GitRestore] style.json restore error: {e}")

    # 3. Restore faq.json — merge auto_learned FAQs
    try:
        remote_faq = _fetch_github_file("knowledge/faq.json")
        if remote_faq:
            remote_data = json.loads(remote_faq)
            local_path = KNOWLEDGE_DIR / "faq.json"
            with open(local_path, "r", encoding="utf-8") as f:
                local_data = json.load(f)

            local_questions = {f["question"].lower() for f in local_data.get("faqs", [])}
            new_faqs = [
                f for f in remote_data.get("faqs", [])
                if f.get("source") == "auto_learned" and f["question"].lower() not in local_questions
            ]
            if new_faqs:
                local_data["faqs"].extend(new_faqs)
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(local_data, f, indent=2, ensure_ascii=False)
                restored.append(f"FAQs: +{len(new_faqs)} learned")
    except Exception as e:
        logger.error(f"[GitRestore] faq.json restore error: {e}")

    # 4. Restore activity_log.json — take the larger log
    try:
        remote_log = _fetch_github_file("knowledge/activity_log.json")
        if remote_log:
            remote_entries = json.loads(remote_log)
            local_path = KNOWLEDGE_DIR / "activity_log.json"

            local_entries = []
            if local_path.exists():
                with open(local_path, "r", encoding="utf-8") as f:
                    local_entries = json.load(f)

            if len(remote_entries) > len(local_entries):
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(remote_entries, f, indent=2, ensure_ascii=False)
                restored.append(f"activity log: {len(local_entries)} → {len(remote_entries)} entries")
    except Exception as e:
        logger.error(f"[GitRestore] activity_log.json restore error: {e}")

    # 5. Restore learned/ files (YouTube knowledge, etc.)
    try:
        learned_files = _list_github_directory("knowledge/learned")
        if learned_files:
            LEARNED_DIR.mkdir(exist_ok=True)
            for file_info in learned_files:
                fname = file_info.get("name", "")
                if fname.startswith("_"):
                    continue  # Skip internal tracking files
                local_file = LEARNED_DIR / fname
                if not local_file.exists():
                    content = _fetch_github_file(f"knowledge/learned/{fname}")
                    if content:
                        with open(local_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        restored.append(f"learned/{fname}")
    except Exception as e:
        logger.error(f"[GitRestore] learned/ restore error: {e}")

    if restored:
        logger.info(f"[GitRestore] Restored {len(restored)} items: {restored}")
    else:
        logger.info("[GitRestore] Everything up to date — nothing to restore")

    return restored


def _fetch_github_file(repo_path: str) -> str | None:
    """Fetch a single file's content from GitHub main branch."""
    url = f"{GITHUB_API}/repos/{settings.github_repo}/contents/{repo_path}"
    try:
        resp = httpx.get(
            url,
            headers=_get_headers(),
            params={"ref": settings.github_branch},
            timeout=15,
        )
        if resp.status_code == 200:
            content_b64 = resp.json().get("content", "")
            return base64.b64decode(content_b64).decode("utf-8")
        return None
    except Exception as e:
        logger.error(f"[GitRestore] Fetch {repo_path} failed: {e}")
        return None


def _list_github_directory(repo_path: str) -> list[dict]:
    """List files in a GitHub directory."""
    url = f"{GITHUB_API}/repos/{settings.github_repo}/contents/{repo_path}"
    try:
        resp = httpx.get(
            url,
            headers=_get_headers(),
            params={"ref": settings.github_branch},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


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
