# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clone, venv setup, and parallel vllm builds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import BuildConfig, Config

# Thread-local storage for log prefix (e.g. "[build 1] ").
_local = threading.local()


def branch_to_dir(branch: str, commit: str = "") -> str:
    """Sanitize branch+commit for use as directory name."""
    d = branch.replace("/", "--")
    if commit:
        d = f"{d}-{commit[:8]}"
    return d


def _prefix() -> str:
    return getattr(_local, "prefix", "")


def _log_file():
    return getattr(_local, "log_file", None)


def _log(msg: str) -> None:
    prefix = _prefix()
    log_f = _log_file()
    for line in msg.splitlines():
        print(f"{prefix}{line}", flush=True)
        if log_f:
            log_f.write(f"{line}\n")


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, prefixing each output line."""
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}
    prefix = _prefix()
    log_f = _log_file()
    cmd_str = " ".join(cmd)
    print(f"{prefix}$ {cmd_str}", flush=True)
    if log_f:
        log_f.write(f"$ {cmd_str}\n")
    if prefix or log_f:
        # Capture output for prefixing and/or logging to file
        result = subprocess.run(
            cmd, cwd=cwd, env=merged_env, check=check,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.stdout:
            for line in result.stdout.splitlines():
                print(f"{prefix}{line}", flush=True)
                if log_f:
                    log_f.write(f"{line}\n")
        return result
    return subprocess.run(
        cmd, cwd=cwd, env=merged_env, check=check,
        stdout=sys.stdout, stderr=sys.stderr,
    )


def clone_or_update(repo_url: str, branch: str, commit: str,
                    repos_dir: Path) -> Path:
    """Clone or fetch+checkout a branch into repos_dir/<sanitized-branch>/."""
    dir_name = branch_to_dir(branch, commit)
    repo_dir = repos_dir / dir_name

    if not repo_dir.exists():
        _log(f"Cloning {repo_url} -> {repo_dir}")
        _run(["git", "clone", repo_url, str(repo_dir)])

    _log(f"Fetching {branch}...")
    _run(["git", "fetch", "origin", branch], cwd=repo_dir)

    if commit:
        _run(["git", "checkout", commit], cwd=repo_dir)
    else:
        _run(["git", "checkout", branch], cwd=repo_dir)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir)

    return repo_dir


def setup_venv(repo_dir: Path) -> None:
    """Create per-repo venv if it doesn't exist."""
    venv_dir = repo_dir / ".venv"
    if not venv_dir.exists():
        _log(f"Creating venv in {repo_dir}...")
        _run(["uv", "venv", "--python", "3.12", "--seed"], cwd=repo_dir)


def _read_build_state(repo_dir: Path) -> dict:
    state_file = repo_dir / ".build_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}


def _write_build_state(repo_dir: Path, state: dict) -> None:
    state_file = repo_dir / ".build_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def _current_build_state(repo_dir: Path, build: BuildConfig) -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
    ).strip()
    return {
        "commit": head,
        "use_precompiled": build.use_precompiled,
        "cuda_arch": build.cuda_arch or "",
        "prebuilt_flash_attn": build.prebuilt_flash_attn,
    }


def build_vllm(repo_dir: Path, build: BuildConfig,
               max_jobs: int | None = None) -> None:
    """Install torch, deps, and build vllm in the repo's venv."""
    current_state = _current_build_state(repo_dir, build)

    # Check if build can be skipped (FORCE_BUILD=1 bypasses cache)
    if os.environ.get("FORCE_BUILD") != "1":
        old_state = _read_build_state(repo_dir)
        if old_state == current_state:
            _log(f"Build already done for "
                 f"{current_state['commit'][:12]} — skipping.")
            return

    venv_python = str(repo_dir / ".venv" / "bin" / "python")
    uv_pip = ["uv", "pip", "install", "--python", venv_python]

    # Install torch + build deps
    _log("Installing torch + build deps...")
    _run(uv_pip + ["torch", "torchvision",
                    "--extra-index-url", build.torch_index])

    # Install build requirements (excluding torch which we just installed)
    build_reqs = repo_dir / "requirements" / "build.txt"
    if build_reqs.exists():
        lines = []
        for raw in build_reqs.read_text().splitlines():
            line = raw.split("#")[0].strip()  # strip inline comments
            if line and not line.startswith("torch=="):
                lines.append(line)
        if lines:
            _run(uv_pip + lines)

    # Optionally install flash-attn
    if build.prebuilt_flash_attn:
        _log("Installing flash-attn (+ build deps)...")
        # flash-attn doesn't declare all build deps; install them first
        _run(uv_pip + ["psutil", "packaging", "ninja"], check=False)
        result = _run(uv_pip + ["flash-attn", "--no-build-isolation"],
                       check=False)
        if result.returncode != 0:
            _log("WARNING: flash-attn install failed, skipping.")

    # Build vllm
    jobs = max_jobs or build.max_jobs
    env: dict[str, str] = {"MAX_JOBS": str(jobs)}

    if build.use_precompiled:
        _log(f"Installing vllm (precompiled, "
             f"HEAD={current_state['commit'][:12]})...")
        env["VLLM_USE_PRECOMPILED"] = "1"
    else:
        _log(f"Building vllm from source "
             f"(HEAD={current_state['commit'][:12]})...")
        if build.cuda_arch:
            env["TORCH_CUDA_ARCH_LIST"] = build.cuda_arch
            cmake_arch = build.cuda_arch.replace(".", "")
            cmake_args = os.environ.get("CMAKE_ARGS", "")
            env["CMAKE_ARGS"] = (
                f"{cmake_args} -DCMAKE_CUDA_ARCHITECTURES={cmake_arch}"
            )

    # Pass through build env vars
    for var in ("VLLM_FLASH_ATTN_SRC_DIR", "VLLM_CUTLASS_SRC_DIR",
                "VLLM_TARGET_DEVICE", "CMAKE_BUILD_TYPE"):
        if os.environ.get(var):
            env[var] = os.environ[var]

    _run(uv_pip + ["-e", ".", "--no-build-isolation"], cwd=repo_dir, env=env)

    _write_build_state(repo_dir, current_state)
    _log("Build complete.")


def _install_one(repo_url: str, build: BuildConfig,
                 branch: str, commit: str,
                 repos_dir: Path, max_jobs: int,
                 builder_id: int = 0,
                 logs_dir: Path | None = None) -> Path:
    """Install a single branch (clone + venv + build). Runs in subprocess."""
    if builder_id:
        _local.prefix = f"[build {builder_id}] "

    dir_name = branch_to_dir(branch, commit)
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        _local.log_file = open(logs_dir / f"build-{dir_name}.log", "w")

    try:
        _log(f"{'=' * 44}")
        _log(f"  Installing: {branch}{f' @ {commit}' if commit else ''}")
        _log(f"{'=' * 44}")

        repo_dir = clone_or_update(repo_url, branch, commit, repos_dir)
        setup_venv(repo_dir)
        build_vllm(repo_dir, build, max_jobs=max_jobs)
        return repo_dir
    finally:
        log_f = _log_file()
        if log_f:
            log_f.close()
            _local.log_file = None


def _unique_branches(runs: list) -> list[tuple[str, str]]:
    """Unique (branch, commit) pairs, preserving order."""
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for run in runs:
        key = (run.branch, run.commit)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def install_all(config: Config,
                repos_dir: Path,
                logs_dir: Path | None = None) -> dict[tuple[str, str], Path]:
    """Build all unique branches, in parallel when possible.

    Returns mapping from (branch, commit) -> repo_dir.
    """
    unique = _unique_branches(config.runs)
    print(f"Installing {len(unique)} unique branch(es)...")

    max_jobs = config.build.max_jobs
    jobs_per_build = max(1, max_jobs // max(1, len(unique)))
    repo_url = config.project.repo
    build = config.build

    if len(unique) == 1:
        branch, commit = unique[0]
        repo_dir = _install_one(repo_url, build, branch, commit,
                                repos_dir, max_jobs=max_jobs,
                                logs_dir=logs_dir)
        return {(branch, commit): repo_dir}

    # Parallel builds
    results: dict[tuple[str, str], Path] = {}
    with ProcessPoolExecutor(max_workers=len(unique)) as pool:
        futures = {
            pool.submit(_install_one, repo_url, build, branch, commit,
                        repos_dir, jobs_per_build,
                        builder_id=i + 1,
                        logs_dir=logs_dir): (branch, commit)
            for i, (branch, commit) in enumerate(unique)
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    return results
