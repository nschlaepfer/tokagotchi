"""Mock API call tool for the agent arena.

Executes HTTP requests against the mock API server running inside the
container at localhost:5000.  Supported endpoints: /weather, /search,
/orders, /orders/<id>.
"""

from __future__ import annotations

import shlex

from src.arena.docker_manager import DockerManager
from src.arena.tools.common import ToolResult

DEFAULT_TIMEOUT = 15
MAX_OUTPUT_CHARS = 10_000
API_BASE = "http://localhost:5000"

# Endpoints the mock server supports
SUPPORTED_ENDPOINTS = {"/weather", "/search", "/orders"}


def _is_supported(endpoint: str) -> bool:
    """Check whether the endpoint is one of the supported mock API routes."""
    normalized = endpoint.rstrip("/")
    if normalized in SUPPORTED_ENDPOINTS:
        return True
    # Allow /orders/<id> pattern
    if normalized.startswith("/orders/"):
        return True
    return False


async def execute(
    docker_mgr: DockerManager,
    container_id: str,
    endpoint: str,
    params: str = "",
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> ToolResult:
    """Execute a curl request against the mock API server in the container.

    Parameters
    ----------
    docker_mgr:
        The Docker manager instance.
    container_id:
        Target container.
    endpoint:
        API endpoint path, e.g. ``/weather`` or ``/orders/ORD-001``.
    params:
        Query parameters as a string, e.g. ``city=london`` or ``q=python``.
        Multiple params separated by ``&``.
    timeout:
        Max seconds to wait.

    Returns
    -------
    ToolResult
        JSON response text from the API, or an error message.
    """
    endpoint = endpoint.strip()
    if not endpoint:
        return ToolResult(
            stdout="",
            stderr="No endpoint specified.",
            exit_code=1,
        )

    # Ensure leading slash
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    if not _is_supported(endpoint):
        return ToolResult(
            stdout="",
            stderr=(
                f"Unsupported endpoint: {endpoint}. "
                f"Available: /weather, /search, /orders, /orders/<id>"
            ),
            exit_code=1,
        )

    # Build the full URL
    url = f"{API_BASE}{endpoint}"
    if params:
        params = params.strip()
        # If params don't start with '?', add the separator
        if not params.startswith("?"):
            url = f"{url}?{params}"
        else:
            url = f"{url}{params}"

    # Use curl with silent mode and JSON content type
    safe_url = shlex.quote(url)
    command = f"curl -s -H 'Content-Type: application/json' {safe_url}"

    try:
        stdout, stderr, exit_code = await docker_mgr.async_exec_in_container(
            container_id, command, timeout=timeout
        )
    except TimeoutError:
        return ToolResult(
            stdout="",
            stderr=f"API call timed out after {timeout}s",
            exit_code=124,
        )

    truncated = False
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = stdout[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        truncated = True

    return ToolResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=truncated,
    )
