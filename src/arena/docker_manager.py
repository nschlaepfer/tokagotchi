"""Manages Docker containers for agent arena episodes.

Provides container lifecycle management, command execution, file transfer,
and a pre-warmed container pool for efficient episode throughput.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
from dataclasses import dataclass, field
from typing import Any

import docker
from docker.errors import APIError, NotFound
from docker.models.containers import Container

from src.models import TaskSpec

logger = logging.getLogger(__name__)

ARENA_IMAGE = "qwen-arena:latest"
WORKSPACE_DIR = "/workspace"
DEFAULT_POOL_SIZE = 4
DEFAULT_EXEC_TIMEOUT = 30


@dataclass
class ContainerPool:
    """Pre-warmed pool of reusable containers."""

    max_size: int = DEFAULT_POOL_SIZE
    _available: list[str] = field(default_factory=list)
    _in_use: set[str] = field(default_factory=set)

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    def acquire(self) -> str | None:
        """Take a container from the pool, or return None if empty."""
        if self._available:
            cid = self._available.pop()
            self._in_use.add(cid)
            return cid
        return None

    def release(self, container_id: str) -> None:
        """Return a container to the pool after cleanup."""
        self._in_use.discard(container_id)
        if len(self._available) < self.max_size:
            self._available.append(container_id)

    def remove(self, container_id: str) -> None:
        """Remove a container from tracking entirely."""
        self._in_use.discard(container_id)
        try:
            self._available.remove(container_id)
        except ValueError:
            pass

    def all_ids(self) -> list[str]:
        return list(self._available) + list(self._in_use)


class DockerManager:
    """Manages Docker containers for the agent arena."""

    def __init__(
        self,
        image: str = ARENA_IMAGE,
        pool_size: int = DEFAULT_POOL_SIZE,
        default_timeout: int = DEFAULT_EXEC_TIMEOUT,
        docker_client: docker.DockerClient | None = None,
    ) -> None:
        self.image = image
        self.default_timeout = default_timeout
        self.client = docker_client or docker.from_env()
        self.pool = ContainerPool(max_size=pool_size)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def warm_pool(self) -> None:
        """Pre-create containers to fill the pool."""
        while self.pool.available_count < self.pool.max_size:
            cid = self._create_raw_container()
            self.pool._available.append(cid)
        logger.info("Pool warmed: %d containers ready", self.pool.available_count)

    async def async_warm_pool(self) -> None:
        await asyncio.to_thread(self.warm_pool)

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def create_container(self, task_spec: TaskSpec) -> str:
        """Create (or reuse) a container for a task, copy initial files in.

        Returns the container ID.
        """
        container_id = self.pool.acquire()
        if container_id is not None:
            logger.info("Reusing pooled container %s", container_id[:12])
            self.cleanup_container(container_id)
        else:
            container_id = self._create_raw_container()
            self.pool._in_use.add(container_id)
            logger.info("Created new container %s", container_id[:12])

        # Copy task files into /workspace
        if task_spec.initial_files:
            self.copy_files_to_container(container_id, task_spec.initial_files)

        return container_id

    async def async_create_container(self, task_spec: TaskSpec) -> str:
        return await asyncio.to_thread(self.create_container, task_spec)

    def _create_raw_container(self) -> str:
        """Spin up a bare container from the arena image."""
        container: Container = self.client.containers.run(
            self.image,
            detach=True,
            stdin_open=True,
            network_mode="bridge",
            mem_limit="512m",
            cpu_period=100_000,
            cpu_quota=50_000,  # 0.5 CPU
            working_dir=WORKSPACE_DIR,
        )
        return container.id

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Execute a command inside the container.

        Returns (stdout, stderr, exit_code).
        Raises TimeoutError if the command exceeds the timeout.
        """
        timeout = timeout or self.default_timeout
        container = self.client.containers.get(container_id)

        exec_handle = self.client.api.exec_create(
            container.id,
            cmd=["bash", "-c", command],
            stdout=True,
            stderr=True,
            workdir=WORKSPACE_DIR,
            user="agent",
        )

        # Use a socket for streaming so we can enforce a timeout
        output = _run_exec_with_timeout(self.client, exec_handle["Id"], timeout)

        inspect = self.client.api.exec_inspect(exec_handle["Id"])
        exit_code: int = inspect.get("ExitCode", -1)

        # Docker muxed stream: split stdout/stderr is complex; for simplicity
        # the combined output is returned as stdout with stderr empty when
        # using combined stream.  For separate streams we would need demux.
        return output, "", exit_code

    async def async_exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        return await asyncio.to_thread(
            self.exec_in_container, container_id, command, timeout
        )

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def copy_files_to_container(
        self,
        container_id: str,
        files: dict[str, str],
    ) -> None:
        """Copy files into the container's /workspace directory.

        Args:
            container_id: Target container.
            files: Mapping of relative path -> file content.
        """
        container = self.client.containers.get(container_id)
        tar_stream = _make_tar(files)
        container.put_archive(WORKSPACE_DIR, tar_stream)

    async def async_copy_files_to_container(
        self,
        container_id: str,
        files: dict[str, str],
    ) -> None:
        await asyncio.to_thread(self.copy_files_to_container, container_id, files)

    # ------------------------------------------------------------------
    # Cleanup / destroy
    # ------------------------------------------------------------------

    def cleanup_container(self, container_id: str) -> None:
        """Reset a container for pool reuse (clear /workspace)."""
        try:
            self.exec_in_container(
                container_id,
                "rm -rf /workspace/* /workspace/.*  2>/dev/null || true",
                timeout=10,
            )
        except Exception:
            logger.warning("Cleanup failed for %s; destroying", container_id[:12])
            self.destroy_container(container_id)

    async def async_cleanup_container(self, container_id: str) -> None:
        await asyncio.to_thread(self.cleanup_container, container_id)

    def destroy_container(self, container_id: str) -> None:
        """Force-remove a container."""
        self.pool.remove(container_id)
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=True)
            logger.info("Destroyed container %s", container_id[:12])
        except NotFound:
            logger.debug("Container %s already removed", container_id[:12])
        except APIError as exc:
            logger.error("Failed to destroy %s: %s", container_id[:12], exc)

    async def async_destroy_container(self, container_id: str) -> None:
        await asyncio.to_thread(self.destroy_container, container_id)

    def release_container(self, container_id: str) -> None:
        """Return a container to the pool (with cleanup) or destroy it."""
        try:
            self.cleanup_container(container_id)
            self.pool.release(container_id)
            logger.info("Released container %s back to pool", container_id[:12])
        except Exception:
            self.destroy_container(container_id)

    async def async_release_container(self, container_id: str) -> None:
        await asyncio.to_thread(self.release_container, container_id)

    def destroy_all(self) -> None:
        """Destroy every tracked container (shutdown)."""
        for cid in self.pool.all_ids():
            self.destroy_container(cid)
        logger.info("All containers destroyed")

    async def async_destroy_all(self) -> None:
        await asyncio.to_thread(self.destroy_all)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_tar(files: dict[str, str]) -> bytes:
    """Create an in-memory tar archive from a filename->content mapping."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def _run_exec_with_timeout(
    client: docker.DockerClient,
    exec_id: str,
    timeout: int,
) -> str:
    """Run a Docker exec and enforce a wall-clock timeout."""
    import concurrent.futures

    def _collect() -> str:
        output = client.api.exec_start(exec_id, stream=False, demux=False)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_collect)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Command exceeded {timeout}s timeout"
            ) from None


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_arena_manager(use_docker: bool = None) -> "DockerManager | SubprocessManager":
    """Create the appropriate arena manager based on Docker availability."""
    if use_docker is None:
        # Auto-detect
        try:
            import docker as _docker
            _docker.from_env().ping()
            use_docker = True
        except Exception:
            use_docker = False

    if use_docker:
        return DockerManager()
    else:
        from src.arena.subprocess_manager import SubprocessManager
        return SubprocessManager()
