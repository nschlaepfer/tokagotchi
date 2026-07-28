#!/usr/bin/env python3
"""Diagnose Tokagotchi dogfood readiness and safety-gate state."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.infra.ollama_utils import ollama_base_urls  # noqa: E402
from src.orchestrator.safety_gates import load_gate_evidence, validate_gate_evidence  # noqa: E402
from scripts.prove_docker_gate import resolve_docker_binary  # noqa: E402
from scripts.prove_docker_gate import DEFAULT_DOCKER_SOCKET  # noqa: E402


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Tokagotchi local readiness.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--json-out", type=Path, help="Optional path to write a JSON report.")
    parser.add_argument("--docker-bin", help="Docker CLI binary to probe.")
    parser.add_argument(
        "--docker-socket",
        default=DEFAULT_DOCKER_SOCKET,
        help="Unix Docker socket required for nested proof validation.",
    )
    parser.add_argument("--skip-git", action="store_true", help="Skip git and gitignore probing.")
    parser.add_argument("--skip-codex", action="store_true", help="Skip Codex CLI probing.")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker CLI/daemon probing.")
    parser.add_argument("--check-ollama", action="store_true", help="Probe local Ollama tags endpoint.")
    parser.add_argument("--probe-timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(main_sync())


def main_sync() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    checks = [
        _python_check(),
        _safety_config_check(cfg),
        _gate_evidence_check(cfg),
    ]
    if args.skip_git:
        checks.append(DoctorCheck("git", "warn", "Git probe skipped by --skip-git."))
    else:
        checks.extend(
            [
                _git_check(),
                _gitignore_check("data/usage_traces"),
                _gitignore_check("data/proofs"),
            ]
        )
    if args.skip_codex:
        checks.append(DoctorCheck("codex_cli", "warn", "Codex CLI probe skipped by --skip-codex."))
    else:
        checks.append(_codex_cli_check(args.probe_timeout_seconds))
    if args.skip_docker:
        checks.append(DoctorCheck("docker", "warn", "Docker probe skipped by --skip-docker."))
    else:
        checks.append(_docker_check(args.docker_bin, args.docker_socket, args.probe_timeout_seconds))
    if args.check_ollama:
        checks.append(_ollama_check(cfg, args.probe_timeout_seconds))

    overall = _overall_status(checks)
    report = {
        "schema_version": 1,
        "overall": overall,
        "repo_root": str(PROJECT_ROOT),
        "checks": [asdict(check) for check in checks],
        "autonomy_locked": any(check.name == "gate_evidence" and check.status != "ok" for check in checks),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _print_report(report)
    return 0 if overall in {"ok", "warn"} else 1


def _python_check() -> DoctorCheck:
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return DoctorCheck("python", "ok", sys.version.split()[0])
    return DoctorCheck("python", "fail", f"Python 3.11+ required, got {sys.version.split()[0]}")


def _git_check() -> DoctorCheck:
    commit = _run(["git", "rev-parse", "HEAD"], timeout=5)
    status = _run(["git", "status", "--short"], timeout=5)
    if commit.returncode != 0:
        return DoctorCheck("git", "fail", "Could not read git commit.")
    dirty = bool(status.stdout.strip())
    detail = f"commit={commit.stdout.strip()[:12]} dirty={dirty}"
    return DoctorCheck("git", "warn" if dirty else "ok", detail)


def _gitignore_check(path: str) -> DoctorCheck:
    candidate = path.rstrip("/") + "/doctor-probe.tmp"
    result = _run(["git", "check-ignore", candidate], timeout=5)
    if result.returncode == 0:
        return DoctorCheck(f"gitignore:{path}", "ok", "ignored")
    return DoctorCheck(f"gitignore:{path}", "fail", "not ignored; generated private/proof data may be committed")


def _codex_cli_check(timeout: float) -> DoctorCheck:
    if shutil.which("codex") is None:
        return DoctorCheck("codex_cli", "fail", "codex CLI not found on PATH")
    result = _run(["codex", "--version"], timeout=timeout)
    if result.returncode == 0 and result.stdout.strip():
        return DoctorCheck("codex_cli", "ok", result.stdout.strip())
    detail = (result.stderr or result.stdout).strip()[:240] or f"exit={result.returncode}"
    return DoctorCheck("codex_cli", "fail", detail)


def _docker_check(configured: str | None, socket_path: str, timeout: float) -> DoctorCheck:
    docker_bin = resolve_docker_binary(configured)
    if not docker_bin:
        return DoctorCheck("docker", "fail", "Docker CLI not found.")
    result = _run([docker_bin, "version"], timeout=timeout)
    if result.returncode == 0:
        if not Path(socket_path).exists():
            return DoctorCheck(
                "docker",
                "fail",
                f"{docker_bin} daemon reachable, but nested proof socket is missing: {socket_path}",
            )
        return DoctorCheck("docker", "ok", f"{docker_bin} daemon reachable")
    if result.returncode == 124:
        return DoctorCheck("docker", "fail", f"{docker_bin} probe timed out after {timeout:g}s")
    detail = (result.stderr or result.stdout).strip()[:240] or f"exit={result.returncode}"
    return DoctorCheck("docker", "fail", f"{docker_bin}: {detail}")


def _ollama_check(cfg: Any, timeout: float) -> DoctorCheck:
    try:
        import requests
    except ImportError:
        return DoctorCheck("ollama", "warn", "requests not installed; skipping Ollama probe")

    errors: list[str] = []
    for api_url in ollama_base_urls(cfg.model.ollama_host, cfg.model.ollama_port):
        try:
            response = requests.get(f"{api_url}/api/tags", timeout=timeout)
            if response.status_code != 200:
                errors.append(f"{api_url}: HTTP {response.status_code}")
                continue
            models = response.json().get("models", [])
            names = {row.get("name") for row in models if isinstance(row, dict)}
            if cfg.model.name in names:
                return DoctorCheck("ollama", "ok", f"{cfg.model.name} available via {api_url}")
            return DoctorCheck("ollama", "warn", f"Ollama reachable via {api_url}, but {cfg.model.name} not listed")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{api_url}: {exc}")
    return DoctorCheck("ollama", "warn", "Ollama not reachable: " + "; ".join(errors))


def _safety_config_check(cfg: Any) -> DoctorCheck:
    enabled = {
        "enable_autonomous_sft": cfg.safety.enable_autonomous_sft,
        "enable_autonomous_rl": cfg.safety.enable_autonomous_rl,
        "enable_checkpoint_promotion": cfg.safety.enable_checkpoint_promotion,
    }
    if any(enabled.values()):
        return DoctorCheck("safety_config", "fail", f"Autonomous gates enabled: {enabled}")
    return DoctorCheck("safety_config", "ok", "autonomous SFT/RL/promotion disabled")


def _gate_evidence_check(cfg: Any) -> DoctorCheck:
    evidence = load_gate_evidence(cfg.safety.gate_evidence_path)
    issues = validate_gate_evidence(evidence)
    if not issues:
        return DoctorCheck("gate_evidence", "ok", f"{cfg.safety.gate_evidence_path} validates")
    return DoctorCheck(
        "gate_evidence",
        "locked",
        f"autonomous learning locked: {', '.join(issues)}",
    )


def _overall_status(checks: list[DoctorCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status in {"warn", "locked"} for check in checks):
        return "warn"
    return "ok"


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=_timeout_text(exc.stdout),
            stderr=_timeout_text(exc.stderr),
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(exc))


def _print_report(report: dict[str, Any]) -> None:
    print(f"overall: {report['overall']}")
    for check in report["checks"]:
        print(f"{check['status']:>6}  {check['name']}: {check['detail']}")
    if report["autonomy_locked"]:
        print("autonomy: locked")


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    main()
