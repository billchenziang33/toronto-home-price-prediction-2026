"""Bounded scan measurements and reversible local exclusions for generated files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

WORKSPACE = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parents[2] / "outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly"
GIT = Path("C:/Program Files/Git/mingw64/bin/git.exe")
PATTERNS = ["**/node_modules/", "**/.venv/", "**/.venv-*/", "**/venv/",
            "**/__pycache__/", "**/.pytest_cache/", "/chrome-profile/",
            "/chrome-profile-2/", "/edge-profile/"]
MARKER = "# Codex desktop responsiveness: generated caches only (2026-10-27)"


def git_args(*args: str) -> list[str]:
    return [str(GIT), "--no-optional-locks", "-c", "safe.bareRepository=explicit",
            "-c", "core.hooksPath=NUL", "-c", "core.fsmonitor=", *args]


def apply() -> dict:
    top = subprocess.run(git_args("rev-parse", "--show-toplevel"), cwd=WORKSPACE,
                         capture_output=True, check=True, text=True).stdout.strip()
    if Path(top).resolve() != WORKSPACE.resolve():
        raise RuntimeError("outer-workspace repository identity differs from the named target")
    path = WORKSPACE / ".git/info/exclude"
    raw = path.read_bytes()
    block = MARKER + "\n" + "\n".join(PATTERNS) + "\n"
    if MARKER in raw.decode("utf-8"):
        raise RuntimeError("desktop-friendly exclusions already exist; inspect instead of replacing")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = OUTPUT / f"outer_git_exclude_before_{stamp}.bak"
    backup.write_bytes(raw)
    path.write_bytes(raw + b"\n" + block.encode("utf-8"))
    result = {"workspace": str(WORKSPACE), "exclude_path": str(path), "backup": str(backup),
              "before_sha256": hashlib.sha256(raw).hexdigest(),
              "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "patterns_added": PATTERNS, "tracked_files_modified": False}
    (OUTPUT / "git_exclusion_change.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def benchmark(label: str) -> dict:
    start = time.monotonic()
    with tempfile.TemporaryFile() as out:
        try:
            done = subprocess.run(git_args("ls-files", "--others", "--exclude-standard", "-z"),
                                  cwd=WORKSPACE, stdout=out, stderr=subprocess.PIPE, timeout=10)
            status, code = "completed", done.returncode
            if code:
                raise RuntimeError(done.stderr.decode("utf-8", errors="replace")[:500])
        except subprocess.TimeoutExpired:
            status, code = "observation_timed_out", None
        size = out.seek(0, 2)
    result = {"label": label, "status": status, "exit_code": code,
              "seconds": time.monotonic() - start, "output_bytes": size,
              "observation_timeout_seconds": 10,
              "command": git_args("ls-files", "--others", "--exclude-standard", "-z"),
              "workspace": str(WORKSPACE)}
    (OUTPUT / f"git_scan_{label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--benchmark", choices=["before", "after"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(apply() if args.apply else benchmark(args.benchmark), indent=2))
