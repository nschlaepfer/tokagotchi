"""Ollama endpoint helpers.

The common dev setup for this repo is Codex/WSL driving a native Windows
Ollama service. In that setup ``localhost`` inside WSL is not necessarily the
same network namespace as Windows, so callers should try the configured host
first and then WSL's gateway addresses.
"""

from __future__ import annotations

import socket
import struct
from pathlib import Path


_LOCALHOST_NAMES = {"localhost", "127.0.0.1", "::1"}


def ollama_base_urls(host: str, port: int) -> list[str]:
    """Return ordered Ollama base URLs for the configured host."""

    candidates = [host]
    if host.lower() in _LOCALHOST_NAMES:
        candidates.extend(_wsl_gateway_hosts())

    urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        url = f"http://{candidate}:{port}"
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def ollama_api_urls(host: str, port: int) -> list[str]:
    """Return ordered native Ollama API URLs."""

    return [f"{base}/api" for base in ollama_base_urls(host, port)]


def _wsl_gateway_hosts() -> list[str]:
    hosts: list[str] = []

    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "nameserver":
                hosts.append(parts[1])
                break

    route = Path("/proc/net/route")
    if route.exists():
        for line in route.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3 and fields[1] == "00000000":
                try:
                    gateway = socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
                except (OSError, ValueError, struct.error):
                    continue
                if gateway != "0.0.0.0":
                    hosts.append(gateway)
                break

    ordered: list[str] = []
    seen: set[str] = set()
    for host in hosts:
        if host not in seen:
            ordered.append(host)
            seen.add(host)
    return ordered
