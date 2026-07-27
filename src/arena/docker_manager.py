"""Manages Docker containers for agent arena episodes.

Provides container lifecycle management, command execution, file transfer,
and a pre-warmed container pool for efficient episode throughput.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import tarfile
from dataclasses import dataclass, field
from typing import Any

try:
    import docker
    from docker.errors import APIError, NotFound
    from docker.models.containers import Container
    _DOCKER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # Docker SDK is optional when using subprocess arena.
    docker = None  # type: ignore[assignment]
    APIError = NotFound = Exception  # type: ignore[misc,assignment]
    Container = Any  # type: ignore[misc,assignment]
    _DOCKER_IMPORT_ERROR = exc

from src.arena.path_safety import workspace_relative_path
from src.models import TaskSpec

logger = logging.getLogger(__name__)

ARENA_IMAGE = "qwen-arena:latest"
WORKSPACE_DIR = "/workspace"
DEFAULT_POOL_SIZE = 4
DEFAULT_EXEC_TIMEOUT = 30


class ArenaUnavailableError(RuntimeError):
    """Raised when no safe arena backend is available."""


class UnsafeArenaBackendError(RuntimeError):
    """Raised when host execution is requested without explicit unsafe opt-in."""


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
        docker_client: Any | None = None,
    ) -> None:
        if docker is None:
            raise RuntimeError(
                "Docker SDK is not installed. Install the project dependencies "
                "or use the subprocess arena backend."
            ) from _DOCKER_IMPORT_ERROR
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
            network_mode="none",
            mem_limit="2g",
            nano_cpus=2_000_000_000,
            pids_limit=64,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            read_only=True,
            tmpfs={WORKSPACE_DIR: "rw,nosuid,nodev,size=512m"},
            working_dir=WORKSPACE_DIR,
            user="agent",
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
        output = _run_exec_with_timeout(
            self.client,
            container_id,
            exec_handle["Id"],
            timeout,
        )

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
        try:
            container.put_archive(WORKSPACE_DIR, tar_stream)
        except APIError as exc:
            if "read-only" not in str(exc).lower():
                raise
            logger.debug(
                "Docker put_archive failed on read-only rootfs; falling back "
                "to in-container workspace writer for %s",
                container_id[:12],
            )
            self._write_files_via_exec(container_id, files)

    async def async_copy_files_to_container(
        self,
        container_id: str,
        files: dict[str, str],
    ) -> None:
        await asyncio.to_thread(self.copy_files_to_container, container_id, files)

    def _write_files_via_exec(self, container_id: str, files: dict[str, str]) -> None:
        """Write task files through Python inside the container.

        Docker's archive API can reject ``put_archive`` when the container uses a
        read-only root filesystem, even when ``/workspace`` is a writable tmpfs.
        This fallback keeps the hardened container settings and writes only
        workspace-relative files from inside the container.
        """

        command = _make_workspace_write_command(files)
        stdout, stderr, exit_code = self.exec_in_container(
            container_id,
            command,
            timeout=max(self.default_timeout, 30),
        )
        if exit_code != 0:
            raise RuntimeError(
                "Failed to write task files into Docker workspace: "
                f"exit={exit_code} stdout={stdout[:500]!r} stderr={stderr[:500]!r}"
            )

    # ------------------------------------------------------------------
    # Cleanup / destroy
    # ------------------------------------------------------------------

    def cleanup_container(self, container_id: str) -> None:
        """Reset a container for pool reuse (clear /workspace)."""
        try:
            self.exec_in_container(
                container_id,
                "find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +",
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
            safe_name = workspace_relative_path(name)
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=safe_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf.read()


def _make_workspace_write_command(files: dict[str, str]) -> str:
    """Build a Python command that writes files under ``/workspace`` safely."""

    payload = {
        workspace_relative_path(name): base64.b64encode(content.encode("utf-8")).decode("ascii")
        for name, content in files.items()
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return (
        "python - <<'PY'\n"
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        f"payload = json.loads({payload_json!r})\n"
        f"root = Path({WORKSPACE_DIR!r}).resolve()\n"
        "for name, encoded in payload.items():\n"
        "    path = (root / name).resolve()\n"
        "    if path != root and root not in path.parents:\n"
        "        raise RuntimeError(f'Unsafe workspace path: {name}')\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_bytes(base64.b64decode(encoded))\n"
        "PY"
    )


def _run_exec_with_timeout(
    client: docker.DockerClient,
    container_id: str,
    exec_id: str,
    timeout: int,
) -> str:
    """Run a Docker exec and enforce a wall-clock timeout."""
    def _collect() -> str:
        output = client.api.exec_start(exec_id, stream=False, demux=False)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_collect)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        try:
            container = client.containers.get(container_id)
            container.remove(force=True)
        except Exception:
            logger.warning("Failed to remove timed-out container %s", container_id[:12], exc_info=True)
        raise TimeoutError(f"Command exceeded {timeout}s timeout") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def create_arena_manager(
    use_docker: bool | None = None,
    *,
    allow_unsafe_host_execution: bool = False,
) -> "DockerManager | SubprocessManager":
    """Create the arena manager.

    Defaults fail closed: Docker unavailable raises instead of silently using
    host subprocess execution. The subprocess backend is available only behind
    an explicit unsafe opt-in for local tests/development.
    """
    if use_docker is None:
        # Auto-detect: ping AND try to list containers (catches credential errors)
        try:
            if docker is None:
                raise RuntimeError("Docker SDK is not installed")
            import docker as _docker
            client = _docker.from_env()
            client.ping()
            client.containers.list(limit=1)  # Actually exercises Docker API
            use_docker = True
            logger.info("Docker detected and working")
        except Exception as e:
            raise ArenaUnavailableError(
                "Docker arena is unavailable and tokagotchi fails closed by default. "
                "Install/start Docker or pass allow_unsafe_host_execution=True only "
                "for explicit local test runs."
            ) from e

    if use_docker:
        return DockerManager()
    if not allow_unsafe_host_execution:
        raise UnsafeArenaBackendError(
            "Subprocess arena executes model-generated commands on the host. "
            "Pass allow_unsafe_host_execution=True only for explicit local test runs."
        )
    from src.arena.subprocess_manager import SubprocessManager
    return SubprocessManager(inherit_environment=False)
