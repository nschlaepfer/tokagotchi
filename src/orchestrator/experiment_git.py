"""Git branch management for experiments.

Provides async git operations for creating experiment branches,
committing results, rolling back to checkpoints, and tagging
best-performing configurations.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentGit:
    """Manages git branches, commits, and tags for the experiment lifecycle.

    Each experiment iteration gets its own branch under ``exp/<loop_id>/``.
    Results are committed with structured messages, and the best checkpoints
    can be tagged for quick retrieval.

    Parameters
    ----------
    repo_path:
        Root directory of the git repository.
    """

    def __init__(self, repo_path: str | Path = ".") -> None:
        self.repo_path = Path(repo_path).resolve()

    # ------------------------------------------------------------------
    # Repository initialisation
    # ------------------------------------------------------------------

    async def init_repo(self, path: str | None = None) -> None:
        """Initialize a git repository if one does not already exist.

        Parameters
        ----------
        path:
            Directory to initialize. Defaults to ``self.repo_path``.
        """
        target = Path(path).resolve() if path else self.repo_path
        git_dir = target / ".git"
        if git_dir.exists():
            logger.info("Git repo already exists at %s", target)
            return

        target.mkdir(parents=True, exist_ok=True)
        await self._run("init", cwd=target)
        # Create an initial commit so branches can be created
        await self._run("commit", "--allow-empty", "-m", "Initial commit", cwd=target)
        logger.info("Initialized git repo at %s", target)

    # ------------------------------------------------------------------
    # Branch management
    # ------------------------------------------------------------------

    async def create_experiment_branch(
        self,
        loop_id: str,
        description: str,
    ) -> str:
        """Create a new experiment branch and check it out.

        The branch name follows the pattern::

            exp/<loop_id>/<YYYYMMDD>_<HHMMSS>_<slug>

        Parameters
        ----------
        loop_id:
            Identifier for the loop (e.g. ``"loop1"``).
        description:
            Human-readable description turned into a filename-safe slug.

        Returns
        -------
        str
            The created branch name.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")[:60]
        branch_name = f"exp/{loop_id}/{timestamp}_{slug}"

        await self._run("checkout", "-b", branch_name)
        logger.info("Created experiment branch: %s", branch_name)
        return branch_name

    # ------------------------------------------------------------------
    # Committing results
    # ------------------------------------------------------------------

    async def commit_results(self, files: list[str], message: str) -> None:
        """Stage the given files and commit them.

        Parameters
        ----------
        files:
            List of file paths (relative to repo root) to stage.
        message:
            Commit message describing the experiment results.
        """
        if not files:
            logger.warning("commit_results called with an empty file list")
            return

        # Stage files
        await self._run("add", "--", *files)
        # Commit (allow empty in case files haven't actually changed)
        await self._run("commit", "-m", message, "--allow-empty")
        logger.info("Committed %d file(s): %s", len(files), message[:80])

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback_to(self, branch_or_commit: str) -> None:
        """Check out a previous branch or commit.

        Parameters
        ----------
        branch_or_commit:
            A branch name, tag, or commit SHA to restore.
        """
        await self._run("checkout", branch_or_commit)
        logger.info("Rolled back to %s", branch_or_commit)

    # ------------------------------------------------------------------
    # Experiment log
    # ------------------------------------------------------------------

    async def get_experiment_log(self, max_entries: int = 50) -> list[dict[str, Any]]:
        """List recent experiment branches with metadata.

        Returns
        -------
        list[dict]
            Each entry contains ``branch``, ``timestamp``, ``message``, and
            ``commit`` (abbreviated SHA).
        """
        # List branches matching the experiment pattern
        stdout = await self._run(
            "for-each-ref",
            "--sort=-creatordate",
            f"--count={max_entries}",
            "--format=%(refname:short)|%(creatordate:iso8601-strict)|%(subject)|%(objectname:short)",
            "refs/heads/exp/",
        )

        entries: list[dict[str, Any]] = []
        for line in stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            entries.append({
                "branch": parts[0],
                "timestamp": parts[1],
                "message": parts[2],
                "commit": parts[3],
            })

        return entries

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    async def tag_best(self, tag_name: str) -> None:
        """Tag the current commit as a best checkpoint.

        Parameters
        ----------
        tag_name:
            Tag name (e.g. ``"best-loop1-v3"``). Overwrites if exists.
        """
        await self._run("tag", "-f", tag_name)
        logger.info("Tagged current commit as %s", tag_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _current_branch(self) -> str:
        """Return the name of the current branch."""
        return (await self._run("rev-parse", "--abbrev-ref", "HEAD")).strip()

    async def _run(
        self,
        *args: str,
        cwd: Path | None = None,
    ) -> str:
        """Execute a git command asynchronously.

        Parameters
        ----------
        args:
            Arguments to ``git`` (e.g. ``"commit"``, ``"-m"``, ``"msg"``).
        cwd:
            Working directory override. Defaults to ``self.repo_path``.

        Returns
        -------
        str
            Combined stdout from the process.

        Raises
        ------
        RuntimeError
            If the git command exits with a non-zero code.
        """
        cmd = ["git", *args]
        work_dir = str(cwd or self.repo_path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (code {proc.returncode}): {stderr.strip()}"
            )

        return stdout
