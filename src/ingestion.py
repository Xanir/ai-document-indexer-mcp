"""
Single-repo ingestion pipeline — top-level folders in the cloned repository
become isolated ChromaDB vector collections.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

log = logging.getLogger(__name__)

# ── Paths (mirror the Docker volume layout) ─────────────────────────────────
CHROMA_DB_PATH = "/data/chroma_db"
REPO_BASE_PATH = "/data/repo"  # single source-of-truth repo

# ── Global singletons (shared across all project collections) ──────────────
_db_client: chromadb.PersistentClient | None = None
_embed_model: FastEmbedEmbedding | None = None


def _get_db_client() -> chromadb.PersistentClient:
    """Lazily initialise the persistent ChromaDB client."""
    global _db_client
    if _db_client is None:
        _db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        log.info("ChromaDB client initialised at %s", CHROMA_DB_PATH)
    return _db_client


def _get_embed_model() -> FastEmbedEmbedding:
    """Lazily initialise the FastEmbed (ONNX) embedding model.

    Controlled via ``FASTEMBED_USE_GPU`` environment variable:

    =============  ====================================================
    Value          Behaviour
    =============  ====================================================
    ``"cuda"``     Force ``CUDAExecutionProvider`` (NVIDIA)
    ``"rocm"``     Force ``ROCmExecutionProvider`` (AMD)
    ``"true"``     Force GPU (tries CUDA first, then ROCm)
    ``"false"``    Force ``CPUExecutionProvider``
    *unset*        Auto — let ONNX Runtime choose the best provider
    =============  ====================================================
    """
    global _embed_model
    if _embed_model is None:
        import os

        use_gpu_env = os.environ.get("FASTEMBED_USE_GPU", "").lower()

        if use_gpu_env == "cuda":
            _embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            log.info(
                "Embedding model loaded: BAAI/bge-small-en-v1.5 "
                "(FastEmbed / ONNX — CUDAExecutionProvider)"
            )
        elif use_gpu_env == "rocm":
            _embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                providers=["ROCmExecutionProvider", "CPUExecutionProvider"],
            )
            log.info(
                "Embedding model loaded: BAAI/bge-small-en-v1.5 "
                "(FastEmbed / ONNX — ROCmExecutionProvider)"
            )
        elif use_gpu_env == "true":
            _embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                providers=[
                    "CUDAExecutionProvider",
                    "ROCmExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
            log.info(
                "Embedding model loaded: BAAI/bge-small-en-v1.5 "
                "(FastEmbed / ONNX — GPU auto-detect)"
            )
        elif use_gpu_env == "false":
            _embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                providers=["CPUExecutionProvider"],
            )
            log.info(
                "Embedding model loaded: BAAI/bge-small-en-v1.5 "
                "(FastEmbed / ONNX — CPU only)"
            )
        else:
            _embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
            log.info(
                "Embedding model loaded: BAAI/bge-small-en-v1.5 "
                "(FastEmbed / ONNX — auto provider selection)"
            )

        Settings.embed_model = _embed_model
    return _embed_model


def get_or_build_project_index(project_name: str) -> VectorStoreIndex:
    """
    Return (or build) the ChromaDB vector index for a top-level folder.

    Each top-level directory under ``/data/repo/`` is treated as an
    independent collection.  If the collection is empty and the folder
    exists on disk, ``.md`` / ``.txt`` files are parsed with
    ``MarkdownNodeParser`` and indexed.
    """
    _get_embed_model()  # ensure Settings.embed_model is populated

    collection = _get_db_client().get_or_create_collection(name=project_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        log.info(
            "Project '%s' already indexed (%d vectors).",
            project_name,
            collection.count(),
        )
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    project_path = Path(REPO_BASE_PATH) / project_name
    if not project_path.exists():
        raise FileNotFoundError(
            f"Folder '{project_name}' not found under {REPO_BASE_PATH}. "
            "Make sure the top-level directory exists in the documentation repo."
        )

    log.info("Indexing markdown files for project '%s' …", project_name)
    documents = SimpleDirectoryReader(
        input_dir=str(project_path),
        required_exts=[".md", ".txt"],
        recursive=True,
    ).load_data()

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    log.info(
        "Project '%s' indexed: %d documents → %d nodes.",
        project_name,
        len(documents),
        len(nodes),
    )
    return index


def drop_collection(project_name: str) -> None:
    """Delete a project's ChromaDB collection so it rebuilds on next search."""
    try:
        _get_db_client().delete_collection(name=project_name)
        log.info("Dropped collection '%s'.", project_name)
    except Exception:
        log.debug("Collection '%s' does not exist — nothing to drop.", project_name)


def get_document_text(project_name: str, file_path: str) -> str:
    """
    Return the full raw text of a single file from the cloned repository.

    *file_path* is relative to the project folder (a top-level directory
    under ``/data/repo/``).
    """
    project_root = Path(REPO_BASE_PATH) / project_name
    full_path = (project_root / file_path).resolve()

    # Safety: ensure the resolved path stays inside the project folder
    if not str(full_path).startswith(str(project_root.resolve())):
        raise ValueError(f"Path traversal detected: {file_path}")

    if not full_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return full_path.read_text(encoding="utf-8")
