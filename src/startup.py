"""
Startup script — hydrates the /data/repo directory from a Git remote
before handing off to the main application entry point.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("startup")

REPO_DIR = Path("/data/repo")


def _auth_url(repo_url: str, pat: str | None) -> str:
    """Embed the PAT as the username in an HTTPS clone URL."""
    if pat and repo_url.startswith("https://"):
        return repo_url.replace("https://", f"https://{pat}@", 1)
    return repo_url


def hydrate_repository() -> None:
    """Clone the repo if it doesn't exist locally, otherwise pull the latest."""
    pat = os.environ.get("GITHUB_PAT")
    repo_url = os.environ.get("REPO_URL")

    if not pat:
        log.warning("GITHUB_PAT is not set — repo must be public or auth will fail.")
    else:
        log.info("GITHUB_PAT loaded (%d chars).", len(pat))
    if not repo_url:
        log.error("REPO_URL environment variable is not set — nothing to clone.")
        sys.exit(1)

    if (REPO_DIR / ".git").exists():
        log.info("Repository already exists at %s — pulling latest changes.", REPO_DIR)
        try:
            #  -c remote.origin.url overrides the URL for this command only.
            #  .git/config stays clean so the host shell can use its own creds.
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO_DIR),
                    "-c",
                    f"remote.origin.url={_auth_url(repo_url, pat)}",
                    "pull",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("Pull complete.")
        except subprocess.CalledProcessError as exc:
            log.error("Git pull failed: %s", exc.stderr.strip())
            sys.exit(1)
    else:
        log.info("Cloning %s into %s ...", repo_url, REPO_DIR)
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", _auth_url(repo_url, pat), str(REPO_DIR)],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("Clone complete.")
        except subprocess.CalledProcessError as exc:
            log.error("Git clone failed: %s", exc.stderr.strip())
            sys.exit(1)


def _clear_and_reindex_all() -> None:
    """Clear all existing ChromaDB collections and rebuild indexes for every
    top-level folder in the cloned repository.

    This ensures the mounted ``/data/chroma_db`` is always populated with
    a fresh index on startup, even if the volume was persisted from a
    previous run or the repo has changed.
    """
    from src.ingestion import (
        CHROMA_DB_PATH,
        REPO_BASE_PATH,
        _get_db_client,
        get_or_build_project_index,
    )

    client = _get_db_client()
    log.info("ChromaDB path: %s", CHROMA_DB_PATH)

    # Drop every existing collection so we always rebuild from scratch.
    try:
        existing = client.list_collections()
        for col in existing:
            client.delete_collection(name=col.name)
            log.info("Cleared existing collection '%s'.", col.name)
    except Exception:
        log.debug("No existing collections to drop.")

    # Enumerate top-level folders and eagerly build their indexes.
    project_dirs = sorted(
        d
        for d in Path(REPO_BASE_PATH).iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not project_dirs:
        log.warning("No project directories found under %s.", REPO_BASE_PATH)
        return

    log.info("Pre-building indexes for %d project(s) …", len(project_dirs))
    for d in project_dirs:
        try:
            get_or_build_project_index(d.name)
        except Exception:
            log.exception("Failed to index project '%s' — continuing.", d.name)

    log.info("All indexes built.")


def main() -> None:
    """Entry point — hydrate the single source-of-truth repo, then start the MCP server."""
    log.info("=== Git hydration starting ===")
    hydrate_repository()
    log.info("=== Git hydration complete ===")

    _clear_and_reindex_all()

    from src.server import mcp

    log.info("Starting MCP server on 0.0.0.0:8000 (SSE transport) …")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
