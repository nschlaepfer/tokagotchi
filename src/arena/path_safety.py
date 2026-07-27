"""Workspace path containment helpers for arena backends and tools."""

from __future__ import annotations

import posixpath
from pathlib import Path


WORKSPACE_ROOT = "/workspace"


def workspace_relative_path(path: str, *, workspace_root: str = WORKSPACE_ROOT) -> str:
    """Return a safe path relative to ``workspace_root``.

    Raises ``ValueError`` when the path is absolute outside the workspace or
    escapes via ``..`` components. This helper is string/posix based because
    Docker paths are Linux paths even when the caller runs from Windows/WSL.
    """

    raw = (path or "").strip()
    if not raw:
        raise ValueError("Path is empty.")

    root = posixpath.normpath(workspace_root)
    if raw.startswith("/"):
        normalized = posixpath.normpath(raw)
    else:
        normalized = posixpath.normpath(posixpath.join(root, raw))

    if normalized == root:
        return "."
    if not normalized.startswith(root + "/"):
        raise ValueError(f"Path escapes {root}: {path}")

    rel = posixpath.relpath(normalized, root)
    if rel == ".." or rel.startswith("../") or rel.startswith("/"):
        raise ValueError(f"Path escapes {root}: {path}")
    return rel


def local_workspace_path(workspace: str | Path, path: str) -> Path:
    """Resolve a workspace-contained path on the local filesystem."""

    workspace_path = Path(workspace).resolve()
    rel = workspace_relative_path(path)
    target = (workspace_path / rel).resolve()
    if target != workspace_path and workspace_path not in target.parents:
        raise ValueError(f"Path escapes workspace: {path}")
    return target
