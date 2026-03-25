"""Custom DSPy LM that routes through the Claude CLI for authentication.

Uses the same ``claude -p`` subprocess approach as opus_client.py,
so it inherits the CLI's auth (no ANTHROPIC_API_KEY needed).

This is used as the ``reflection_lm`` for DSPy GEPA, allowing Opus
to analyze traces and propose mutations without a separate API key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_claude_cli() -> list[str]:
    """Find the Claude CLI executable, bypassing .CMD wrapper on Windows."""
    import shutil
    import os

    # Known paths for Windows npm global installs
    npm_prefixes = [
        os.path.expandvars(r"%APPDATA%\npm"),
        os.path.expanduser("~/AppData/Roaming/npm"),
        "C:/Users/" + os.environ.get("USERNAME", "") + "/AppData/Roaming/npm",
    ]

    # Also try `npm prefix -g` if npm is available
    try:
        result = subprocess.run(
            ["npm", "prefix", "-g"],
            capture_output=True, text=True, timeout=10,
            shell=True,  # Needed on Windows to find npm.cmd
        )
        if result.returncode == 0 and result.stdout.strip():
            npm_prefixes.insert(0, result.stdout.strip())
    except Exception:
        pass

    # Look for cli.js in known locations
    for prefix in npm_prefixes:
        prefix = prefix.replace("\\", "/")
        cli_js = Path(prefix) / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
        if cli_js.exists():
            node = shutil.which("node") or "node"
            return [node, str(cli_js)]

    # Fallback to `claude` on PATH (may be .CMD wrapper)
    claude = shutil.which("claude")
    if claude:
        # If it's a .cmd, use node + cli.js instead
        if claude.lower().endswith(".cmd"):
            # Parse the .cmd to find the real path
            try:
                with open(claude) as f:
                    content = f.read()
                # .cmd typically contains: @"%~dp0\node_modules\...\cli.js" %*
                import re
                match = re.search(r'node_modules[\\/@\w.-]+cli\.js', content)
                if match:
                    cli_path = Path(claude).parent / match.group()
                    if cli_path.exists():
                        node = shutil.which("node") or "node"
                        return [node, str(cli_path)]
            except Exception:
                pass
        return [claude]

    raise FileNotFoundError("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")


def call_claude_cli(prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude Opus via the CLI and return the text response.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    max_tokens : int
        Not directly used (CLI manages this), but kept for interface compat.

    Returns
    -------
    str
        The model's text response.
    """
    cli_cmd = _resolve_claude_cli()

    if len(prompt) > 4000:
        cmd = [
            *cli_cmd,
            "-p", "-",
            "--output-format", "json",
            "--max-turns", "1",
            "--allowedTools", "",
        ]
        stdin_text = prompt
    else:
        cmd = [
            *cli_cmd,
            "-p", prompt,
            "--output-format", "json",
            "--max-turns", "1",
            "--allowedTools", "",
        ]
        stdin_text = None

    try:
        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning("Claude CLI returned code %d: %s", result.returncode, result.stderr[:200])
            return f"Error: CLI returned code {result.returncode}"

        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            return "Error: Empty response from CLI"

        try:
            data = json.loads(output)
            # Extract text from JSON response
            if isinstance(data, dict):
                # Claude CLI JSON format: {"result": "text", ...} or {"content": [...]}
                if "result" in data:
                    return data["result"]
                if "content" in data:
                    parts = data["content"]
                    if isinstance(parts, list):
                        return " ".join(
                            p.get("text", "") for p in parts if isinstance(p, dict)
                        )
                    return str(parts)
                # Fallback: return the whole thing as text
                return json.dumps(data)
            return str(data)
        except json.JSONDecodeError:
            # Not JSON — return raw text
            return output

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after 120s")
        return "Error: CLI timed out"
    except Exception as e:
        logger.error("Claude CLI call failed: %s", e)
        return f"Error: {e}"


class ClaudeCliLM:
    """A DSPy-compatible LM that routes through the Claude CLI.

    This is NOT a full dspy.LM subclass — it's a minimal adapter that
    DSPy GEPA can use as a reflection LM. DSPy calls
    ``lm(prompt, **kwargs)`` and expects a list of string completions.

    Usage::

        reflection_lm = ClaudeCliLM()
        optimizer = dspy.GEPA(reflection_lm=reflection_lm, ...)
    """

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self.model = model
        self.history: list[dict[str, Any]] = []

    def __call__(self, prompt: str | None = None, messages: list | None = None, **kwargs: Any) -> list[str]:
        """Call the Claude CLI and return completions.

        DSPy's internal calls use either positional prompt or messages kwarg.
        """
        if messages:
            # Convert messages to a single prompt string
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                parts.append(f"[{role}]: {content}")
            prompt_text = "\n\n".join(parts)
        elif prompt:
            prompt_text = prompt
        else:
            return [""]

        response = call_claude_cli(prompt_text)
        self.history.append({
            "prompt": prompt_text[:200],
            "response": response[:200],
        })
        return [response]

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        """Return the last n calls for debugging."""
        return self.history[-n:]

    @property
    def kwargs(self) -> dict[str, Any]:
        """DSPy accesses this for model metadata."""
        return {"model": self.model, "temperature": 1.0, "max_tokens": 4096}
