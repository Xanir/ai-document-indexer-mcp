"""
FastMCP server — exposes the document indexer as MCP tools over SSE transport.
"""

import os
import subprocess

from fastmcp import FastMCP

from .ingestion import (
    REPO_BASE_PATH,
    drop_collection,
    get_document_text,
    get_or_build_project_index,
)

mcp = FastMCP("ProjectKnowledgeBase")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _git_pull() -> subprocess.CompletedProcess[str]:
    """Pull using the PAT-injected URL for this command only.

    ``-c remote.origin.url=…`` overrides the remote for a single
    invocation without modifying ``.git/config``.  The host shell
    (which shares the repo via bind-mount) keeps a clean URL and
    can use its own credentials for manual commits and pushes.
    """
    repo_url = os.environ.get("REPO_URL", "")
    pat = os.environ.get("GITHUB_PAT")

    cmd = ["git", "-C", REPO_BASE_PATH]
    if pat and repo_url.startswith("https://"):
        auth_url = repo_url.replace("https://", f"https://{pat}@", 1)
        cmd.extend(["-c", f"remote.origin.url={auth_url}"])
    cmd.append("pull")

    return subprocess.run(cmd, check=True, capture_output=True, text=True)


# ── MCP Tools ───────────────────────────────────────────────────────────────


@mcp.tool()
def search_architecture(project_name: str, query: str) -> str:
    """
    Search the isolated knowledge base for a specific project.

    Always use this to understand design decisions, architecture, or
    project requirements captured in the indexed markdown files.
    """
    try:
        index = get_or_build_project_index(project_name)
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)

        context = f"--- Retrieved Context for {project_name} ---\n\n"
        for node in nodes:
            file_name = node.metadata.get("file_name", "Unknown File")
            context += f"Source: {file_name}\n{node.text}\n\n"

        return context
    except Exception as exc:
        return f"Error searching architecture: {exc}"


@mcp.tool()
def get_document_context(project_name: str, file_path: str) -> str:
    """
    Fetch the full raw text of a specific markdown file from the cloned
    repository, bypassing the vector search.  Useful when the AI needs the
    exact, unabridged content of a known file.
    """
    try:
        return get_document_text(project_name, file_path)
    except Exception as exc:
        return f"Error reading document: {exc}"


@mcp.tool()
def sync_knowledge_base() -> str:
    """
    Pull the latest documentation from the single source-of-truth repo
    and clear cached indexes so they rebuild on the next search.

    Call this after pushing documentation changes upstream.
    """
    try:
        result = _git_pull()

        # Drop cached collections so they rebuild against the new files
        # on the next search_architecture call.
        for entry in sorted(fs for fs in os.scandir(REPO_BASE_PATH) if fs.is_dir()):
            drop_collection(entry.name)

        return (
            "Pulled latest documentation successfully.\n\n"
            f"git output:\n{result.stdout.strip()}"
        )
    except subprocess.CalledProcessError as exc:
        return f"Git pull failed: {exc.stderr.strip()}"
