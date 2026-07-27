# Docker Proof Gate

Updated: 2026-07-27

This is the machine proof step before autonomous SFT, RL, or checkpoint promotion can be considered.

Do not unlock autonomous learning from a host-only run. Host proof is useful for development, but the gate requires Docker-backed evidence.

## What the proof does

`scripts/prove_docker_gate.py`:

1. runs `git diff --check` on the host;
2. checks the Docker daemon;
3. builds the arena image from `docker/Dockerfile.arena`;
4. builds a clean proof image from `docker/Dockerfile.proof`;
5. runs validation inside the proof image;
6. mounts the host Unix Docker socket into that proof image to run the arena-backed executable task-bank checks;
7. writes JSON logs, task-bank reports, integration-test reports, and a `truth_gate_candidate.json`.

The generated candidate keeps:

```json
"human_reviewed": false
```

That is intentional. A human must inspect the proof artifacts before any evidence file is promoted into `data/safety/truth_gate.json`.

## Run it

From the repo root:

```bash
python3 scripts/prove_docker_gate.py
```

Expected output:

```text
Docker proof passed. Artifact: data/proofs/docker_gate/<timestamp>/proof.json
Truth-gate candidate: data/proofs/docker_gate/<timestamp>/truth_gate_candidate.json
```

Generated artifacts are ignored by Git under `data/proofs/`.

Installed entry point:

```bash
tokagotchi-prove-docker-gate
```

Check readiness first:

```bash
python3 scripts/tokagotchi_doctor.py --check-ollama
```

Inside CI/proof containers, do not require user-local tools such as a logged-in
Codex CLI or local Ollama service. Use explicit skip flags for that environment:

```bash
python3 scripts/tokagotchi_doctor.py --skip-codex --skip-docker
```

## WSL with Windows Docker Desktop

If WSL integration is not enabled but Docker Desktop is installed on Windows, you can point the runner at `docker.exe`:

```bash
python3 scripts/prove_docker_gate.py \
  --docker-bin "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
```

Docker Desktop must be running, the Linux engine must be healthy, and a Unix Docker socket must be available to mount into the proof container. In practice, enabling Docker Desktop WSL integration is the cleanest local path.

If only `docker.exe` is present but `/var/run/docker.sock` is missing, the runner records a blocked proof instead of unlocking anything.

## GitHub Actions proof

This repo includes `.github/workflows/docker-proof.yml`. On GitHub-hosted Ubuntu runners, Docker and `/var/run/docker.sock` are available by default, so the workflow can run the proof gate without depending on local Docker Desktop state.

The workflow uploads `data/proofs/docker_gate/` as an artifact. The generated `truth_gate_candidate.json` still requires human review before any local safety flags are changed.

The proof runner sets `TOKAGOTCHI_SKIP_EXTERNAL_PROBES=1` inside the proof
container so the integration suite verifies local wiring and safety gates
without waiting on user-local Ollama/Codex services that GitHub Actions should
not possess.

## Dry-run the command plan

Use this where Docker is unavailable:

```bash
python3 scripts/prove_docker_gate.py --dry-run
```

This writes a `proof.json` showing the exact Docker commands that would run, but it does not satisfy the gate.

## What must pass

The proof is only acceptable when all of these are true:

- Docker daemon is available.
- `/var/run/docker.sock` or the configured `--docker-socket` path exists and can be mounted into the proof container.
- Arena image builds successfully.
- Proof image builds successfully.
- The proof container exits successfully.
- `python -m compileall -q src scripts` passes inside the proof image.
- Static task-bank validation passes.
- Docker-backed executable task-bank validation passes.
- `scripts/test_all_loops.py --json-out ...` passes with zero failures.
- The git tree is clean.

The current safety gate also requires `git_dirty: false` in the evidence file.

## Review before promotion

Before copying or adapting `truth_gate_candidate.json` into `data/safety/truth_gate.json`, inspect:

- `proof.json`
- `truth_gate_candidate.json`
- `inside-results.json`
- `integration_tests.json`
- `task_bank_static.json`
- `task_bank_docker.json`
- `logs/*.log`

Only after review should the evidence be marked `human_reviewed: true`.

Do not change the safety config flags unless the reviewed evidence validates cleanly.
