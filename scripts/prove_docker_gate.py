#!/usr/bin/env python3
"""Generate reproducible Docker-backed proof for Tokagotchi safety gates.

The default mode runs from the host and requires Docker. It builds the arena
image, builds a clean proof image from the current checkout, runs validation in
that proof image, and writes machine-checkable artifacts under data/proofs.

The generated truth-gate file is intentionally a candidate:
``human_reviewed`` is false. A human must inspect the artifacts before copying
or editing it into config.safety.gate_evidence_path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "proofs" / "docker_gate"
DEFAULT_TASK_BANK = Path("data/curriculum/seed_tasks.json")
DEFAULT_ARENA_IMAGE = "qwen-arena:latest"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_DOCKER_SOCKET = "/var/run/docker.sock"


@dataclass
class CommandRecord:
    name: str
    command: str
    exit_code: int
    duration_seconds: float
    stdout_log: str = ""
    stderr_log: str = ""
    timed_out: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Docker-backed Tokagotchi proof and write artifacts."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-bank", type=Path, default=DEFAULT_TASK_BANK)
    parser.add_argument("--arena-image", default=DEFAULT_ARENA_IMAGE)
    parser.add_argument(
        "--proof-image",
        help="Docker image tag for the proof runner. Defaults to tokagotchi-proof:<short-sha>.",
    )
    parser.add_argument(
        "--docker-bin",
        default=os.environ.get("TOKAGOTCHI_DOCKER_BIN"),
        help=(
            "Docker CLI binary to use. Defaults to TOKAGOTCHI_DOCKER_BIN, "
            "then docker on PATH. From WSL you can point this at Windows "
            "docker.exe when Docker Desktop is running."
        ),
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout for each host/proof command.",
    )
    parser.add_argument(
        "--docker-probe-timeout-seconds",
        type=int,
        default=30,
        help="Short timeout for Docker daemon probes before builds/tests start.",
    )
    parser.add_argument(
        "--docker-socket",
        default=os.environ.get("TOKAGOTCHI_DOCKER_SOCKET", DEFAULT_DOCKER_SOCKET),
        help=(
            "Unix Docker socket to mount into the proof container for nested "
            "arena validation. Defaults to /var/run/docker.sock."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the command plan without running Docker.",
    )
    parser.add_argument(
        "--inside-container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--proof-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inside_container:
        if args.proof_dir is None:
            raise SystemExit("--inside-container requires --proof-dir")
        raise SystemExit(run_inside_container(args))
    raise SystemExit(run_from_host(args))


def run_from_host(args: argparse.Namespace) -> int:
    proof_dir = _new_proof_dir(args.output_root)
    logs_dir = proof_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _git_text(["git", "rev-parse", "HEAD"]) or "unknown"
    git_status = _git_text(["git", "status", "--short"])
    git_dirty = bool(git_status.strip())
    short_sha = _safe_tag(git_commit[:12] if git_commit != "unknown" else "unknown")
    proof_image = args.proof_image or f"tokagotchi-proof:{short_sha}"
    task_bank = _posix_relpath(args.task_bank)
    docker_bin = resolve_docker_binary(args.docker_bin)

    command_plan = _host_command_plan(
        proof_dir=proof_dir,
        docker_bin=docker_bin or "docker",
        docker_socket=args.docker_socket,
        proof_image=proof_image,
        arena_image=args.arena_image,
        task_bank=task_bank,
        timeout_seconds=args.command_timeout_seconds,
    )

    if args.dry_run:
        proof = {
            "schema_version": 1,
            "status": "not_run",
            "reason": "dry_run",
            "generated_at": _utc_now(),
            "repo_root": str(PROJECT_ROOT),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "proof_dir": str(proof_dir),
            "command_plan": [shlex.join(cmd) for cmd in command_plan],
        }
        _write_json(proof_dir / "proof.json", proof)
        print(f"Dry-run proof plan written to {proof_dir / 'proof.json'}")
        return 0

    records: list[CommandRecord] = []
    records.append(
        run_command(
            "git_diff_check",
            ["git", "diff", "--check"],
            cwd=PROJECT_ROOT,
            logs_dir=logs_dir,
            timeout_seconds=args.command_timeout_seconds,
        )
    )

    if not docker_bin:
        proof = _blocked_proof(
            proof_dir=proof_dir,
            git_commit=git_commit,
            git_dirty=git_dirty,
            reason="docker_cli_not_found",
            records=records,
            command_plan=command_plan,
        )
        _write_json(proof_dir / "proof.json", proof)
        print(f"Docker proof blocked: docker CLI not found. Artifact: {proof_dir / 'proof.json'}")
        return 2

    records.append(
        run_command(
            "docker_version",
            [docker_bin, "version"],
            cwd=PROJECT_ROOT,
            logs_dir=logs_dir,
            timeout_seconds=args.docker_probe_timeout_seconds,
        )
    )
    if records[-1].exit_code != 0:
        proof = _blocked_proof(
            proof_dir=proof_dir,
            git_commit=git_commit,
            git_dirty=git_dirty,
            reason="docker_daemon_unavailable",
            records=records,
            command_plan=command_plan,
        )
        _write_json(proof_dir / "proof.json", proof)
        print(f"Docker proof blocked: Docker daemon unavailable. Artifact: {proof_dir / 'proof.json'}")
        return 2

    socket_path = Path(args.docker_socket)
    if not socket_path.exists():
        proof = _blocked_proof(
            proof_dir=proof_dir,
            git_commit=git_commit,
            git_dirty=git_dirty,
            reason="docker_socket_unavailable_for_nested_proof",
            records=records,
            command_plan=command_plan,
        )
        _write_json(proof_dir / "proof.json", proof)
        print(
            "Docker proof blocked: Docker socket unavailable for nested arena "
            f"validation at {socket_path}. Artifact: {proof_dir / 'proof.json'}"
        )
        return 2

    for name, command in (
        (
            "build_arena_image",
            [
                docker_bin,
                "build",
                "-t",
                args.arena_image,
                "-f",
                "docker/Dockerfile.arena",
                "docker",
            ],
        ),
        (
            "build_proof_image",
            [
                docker_bin,
                "build",
                "-t",
                proof_image,
                "-f",
                "docker/Dockerfile.proof",
                ".",
            ],
        ),
    ):
        record = run_command(
            name,
            command,
            cwd=PROJECT_ROOT,
            logs_dir=logs_dir,
            timeout_seconds=args.command_timeout_seconds,
        )
        records.append(record)
        if record.exit_code != 0:
            proof = _blocked_proof(
                proof_dir=proof_dir,
                git_commit=git_commit,
                git_dirty=git_dirty,
                reason=f"{name}_failed",
                records=records,
                command_plan=command_plan,
            )
            _write_json(proof_dir / "proof.json", proof)
            print(f"Docker proof failed during {name}. Artifact: {proof_dir / 'proof.json'}")
            return 1

    docker_run = command_plan[-1]
    records.append(
        run_command(
            "docker_proof_container",
            docker_run,
            cwd=PROJECT_ROOT,
            logs_dir=logs_dir,
            timeout_seconds=args.command_timeout_seconds,
        )
    )

    inside = _read_json(proof_dir / "inside-results.json")
    gate_candidate = build_truth_gate_candidate(
        proof_dir=proof_dir,
        git_commit=git_commit,
        git_dirty=git_dirty,
        arena_image=args.arena_image,
        proof_image=proof_image,
        task_bank=task_bank,
        host_records=records,
        inside=inside,
    )
    _write_json(proof_dir / "truth_gate_candidate.json", gate_candidate)

    proof_status = "passed" if gate_candidate["truth_grounding_passed"] else "failed"
    proof = {
        "schema_version": 1,
        "status": proof_status,
        "generated_at": _utc_now(),
        "repo_root": str(PROJECT_ROOT),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "proof_dir": str(proof_dir),
        "arena_image": args.arena_image,
        "proof_image": proof_image,
        "commands": [asdict(r) for r in records],
        "inside": inside,
        "truth_gate_candidate": str(proof_dir / "truth_gate_candidate.json"),
        "note": "human_reviewed is false by design; review before enabling autonomous gates.",
    }
    _write_json(proof_dir / "proof.json", proof)
    print(f"Docker proof {proof_status}. Artifact: {proof_dir / 'proof.json'}")
    print(f"Truth-gate candidate: {proof_dir / 'truth_gate_candidate.json'}")
    return 0 if proof_status == "passed" else 1


def run_inside_container(args: argparse.Namespace) -> int:
    proof_dir = args.proof_dir
    logs_dir = proof_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    task_bank = str(args.task_bank)
    commands = [
        (
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "scripts"],
        ),
        (
            "task_bank_static",
            [
                sys.executable,
                "scripts/validate_task_bank.py",
                task_bank,
                "--static-only",
                "--summary",
                "--json-out",
                str(proof_dir / "task_bank_static.json"),
            ],
        ),
        (
            "task_bank_docker",
            [
                sys.executable,
                "scripts/validate_task_bank.py",
                task_bank,
                "--summary",
                "--json-out",
                str(proof_dir / "task_bank_docker.json"),
            ],
        ),
        (
            "integration_suite",
            [
                sys.executable,
                "scripts/test_all_loops.py",
                "--json-out",
                str(proof_dir / "integration_tests.json"),
            ],
        ),
    ]

    records: list[CommandRecord] = []
    for name, command in commands:
        record = run_command(
            name,
            command,
            cwd=PROJECT_ROOT,
            logs_dir=logs_dir,
            timeout_seconds=args.command_timeout_seconds,
        )
        records.append(record)
        if record.exit_code != 0:
            break

    inside = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "commands": [asdict(r) for r in records],
        "task_bank_static": _read_json(proof_dir / "task_bank_static.json"),
        "task_bank_docker": _read_json(proof_dir / "task_bank_docker.json"),
        "integration_tests": _read_json(proof_dir / "integration_tests.json"),
    }
    inside["success"] = all(r.exit_code == 0 for r in records) and len(records) == len(commands)
    _write_json(proof_dir / "inside-results.json", inside)
    return 0 if inside["success"] else 1


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    logs_dir: Path,
    timeout_seconds: int,
) -> CommandRecord:
    start = time.monotonic()
    stdout_log = logs_dir / f"{name}.stdout.log"
    stderr_log = logs_dir / f"{name}.stderr.log"
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        return CommandRecord(
            name=name,
            command=shlex.join(command),
            exit_code=completed.returncode,
            duration_seconds=time.monotonic() - start,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_log.write_text(_timeout_text(exc.stdout), encoding="utf-8")
        stderr_log.write_text(_timeout_text(exc.stderr), encoding="utf-8")
        return CommandRecord(
            name=name,
            command=shlex.join(command),
            exit_code=124,
            duration_seconds=time.monotonic() - start,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            timed_out=True,
        )


def build_truth_gate_candidate(
    *,
    proof_dir: Path,
    git_commit: str,
    git_dirty: bool,
    arena_image: str,
    proof_image: str,
    task_bank: str,
    host_records: list[CommandRecord],
    inside: dict[str, Any],
) -> dict[str, Any]:
    static_report = inside.get("task_bank_static") if isinstance(inside, dict) else {}
    docker_report = inside.get("task_bank_docker") if isinstance(inside, dict) else {}
    tests = inside.get("integration_tests") if isinstance(inside, dict) else {}

    static_summary = summarize_task_bank(static_report)
    docker_summary = summarize_task_bank(docker_report)
    test_summary = summarize_tests(tests)
    inside_commands = [
        c for c in inside.get("commands", [])
        if isinstance(c, dict) and c.get("command")
    ] if isinstance(inside, dict) else []
    host_git_diff = next((r for r in host_records if r.name == "git_diff_check"), None)

    reproducible_commands: list[dict[str, Any]] = []
    if host_git_diff is not None:
        reproducible_commands.append(
            {"command": host_git_diff.command, "exit_code": host_git_diff.exit_code}
        )
    for command in inside_commands:
        reproducible_commands.append(
            {
                "command": str(command.get("command", "")),
                "exit_code": int(command.get("exit_code", -1)),
            }
        )

    machine_passed = (
        all(record.exit_code == 0 for record in host_records)
        and bool(inside.get("success"))
        and static_summary.get("valid") is True
        and docker_summary.get("valid") is True
        and test_summary.get("failures") == 0
        and git_dirty is False
    )

    return {
        "schema_version": 1,
        "truth_grounding_passed": machine_passed,
        "human_reviewed": False,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "task_judge_canonical": True,
        "arena": {
            "backend": "docker",
            "network": "none",
            "fail_closed_checked": test_summary.get("failures") == 0,
            "unsafe_host_execution": False,
            "arena_image": arena_image,
            "proof_image": proof_image,
        },
        "task_bank": {
            "path": task_bank,
            "static_valid": static_summary.get("valid") is True,
            "executable_valid": docker_summary.get("valid") is True,
            "tasks": docker_summary.get("tasks") or static_summary.get("tasks") or 0,
            "executable_tasks": docker_summary.get("executable_tasks") or 0,
            "starters_failed": docker_summary.get("starters_failed") or 0,
            "references_passed": docker_summary.get("references_passed") or 0,
            "benchmark_tasks": docker_summary.get("benchmark_tasks") or 0,
            "benchmarks_passed": docker_summary.get("benchmarks_passed") or 0,
            "invalid_task_ids": sorted(
                set(static_summary.get("invalid_task_ids") or [])
                | set(docker_summary.get("invalid_task_ids") or [])
            ),
        },
        "tests": test_summary,
        "reproducible_commands": reproducible_commands,
        "proof_artifacts": {
            "proof_dir": str(proof_dir),
            "inside_results": str(proof_dir / "inside-results.json"),
            "integration_tests": str(proof_dir / "integration_tests.json"),
            "task_bank_static": str(proof_dir / "task_bank_static.json"),
            "task_bank_docker": str(proof_dir / "task_bank_docker.json"),
        },
    }


def summarize_tests(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "suite": "scripts/test_all_loops.py",
            "total": 0,
            "passed": 0,
            "failures": 1,
            "skipped": 0,
        }
    return {
        "suite": report.get("suite", "scripts/test_all_loops.py"),
        "total": int(report.get("total") or 0),
        "passed": int(report.get("passed") or 0),
        "failures": int(report.get("failures") or 0),
        "skipped": int(report.get("skipped") or 0),
    }


def summarize_task_bank(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "tasks": 0,
            "valid": False,
            "executable_tasks": 0,
            "starters_failed": 0,
            "references_passed": 0,
            "benchmark_tasks": 0,
            "benchmarks_passed": 0,
            "invalid_task_ids": [],
        }
    results = report.get("results") or []
    executable = [
        row for row in results
        if isinstance(row, dict)
        and (row.get("starter_result") is not None or row.get("reference_result") is not None)
    ]
    starters_failed = sum(
        1 for row in executable
        if (row.get("starter_result") or {}).get("oracle_passed") is False
    )
    references_passed = sum(
        1 for row in executable
        if (row.get("reference_result") or {}).get("success") is True
    )
    benchmark = [
        row for row in executable
        if (row.get("reference_result") or {}).get("task_type") == "open_ended_optimization"
    ]
    benchmarks_passed = sum(
        1 for row in benchmark
        if (row.get("reference_result") or {}).get("details", {}).get("benchmark_passed") is True
    )
    invalid = [
        str(row.get("task_id"))
        for row in results
        if isinstance(row, dict) and row.get("valid") is not True
    ]
    return {
        "tasks": int(report.get("tasks") or len(results)),
        "valid": report.get("valid") is True,
        "executable_tasks": len(executable),
        "starters_failed": starters_failed,
        "references_passed": references_passed,
        "benchmark_tasks": len(benchmark),
        "benchmarks_passed": benchmarks_passed,
        "invalid_task_ids": invalid,
    }


def _host_command_plan(
    *,
    proof_dir: Path,
    docker_bin: str,
    docker_socket: str,
    proof_image: str,
    arena_image: str,
    task_bank: str,
    timeout_seconds: int,
) -> list[list[str]]:
    return [
        ["git", "diff", "--check"],
        [docker_bin, "version"],
        [docker_bin, "build", "-t", arena_image, "-f", "docker/Dockerfile.arena", "docker"],
        [docker_bin, "build", "-t", proof_image, "-f", "docker/Dockerfile.proof", "."],
        [
            docker_bin,
            "run",
            "--rm",
            "-v",
            f"{docker_socket}:/var/run/docker.sock",
            "-v",
            f"{proof_dir}:/proof",
            proof_image,
            "python",
            "scripts/prove_docker_gate.py",
            "--inside-container",
            "--proof-dir",
            "/proof",
            "--task-bank",
            task_bank,
            "--command-timeout-seconds",
            str(timeout_seconds),
        ],
    ]


def _blocked_proof(
    *,
    proof_dir: Path,
    git_commit: str,
    git_dirty: bool,
    reason: str,
    records: list[CommandRecord],
    command_plan: list[list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "blocked",
        "reason": reason,
        "generated_at": _utc_now(),
        "repo_root": str(PROJECT_ROOT),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "proof_dir": str(proof_dir),
        "commands": [asdict(r) for r in records],
        "command_plan": [shlex.join(cmd) for cmd in command_plan],
    }


def resolve_docker_binary(configured: str | None) -> str | None:
    """Resolve the Docker CLI binary, including common WSL + Docker Desktop paths."""

    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    path_docker = shutil.which("docker")
    if path_docker:
        sibling_exe = Path(path_docker).with_suffix(".exe")
        if sibling_exe.exists():
            candidates.append(str(sibling_exe))
        candidates.append(path_docker)
    windows_docker = Path("/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe")
    if windows_docker.exists():
        candidates.append(str(windows_docker))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    return None


def _new_proof_dir(root: Path) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    proof_dir = root / stamp
    counter = 1
    while proof_dir.exists():
        proof_dir = root / f"{stamp}-{counter}"
        counter += 1
    proof_dir.mkdir(parents=True, exist_ok=False)
    return proof_dir


def _git_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _posix_relpath(path: Path) -> str:
    if path.is_absolute():
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _safe_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value or "unknown"


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
