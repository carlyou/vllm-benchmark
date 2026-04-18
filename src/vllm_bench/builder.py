# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clone, venv setup, and vllm builds."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import BuildConfig, Config


@dataclass
class BuildContext:
    """Explicit context for build logging (replaces thread-local state)."""

    prefix: str = ""
    log_file: IO | None = None

    def log(self, msg: str) -> None:
        for line in msg.splitlines():
            print(f"{self.prefix}{line}", flush=True)
            if self.log_file:
                self.log_file.write(f"{line}\n")


def branch_to_dir(branch: str, commit: str = "") -> str:
    """Sanitize branch+commit for use as directory name."""
    d = branch.replace("/", "--")
    if commit:
        d = f"{d}-{commit[:8]}"
    return d


def _resolve_jobs(value: float) -> int:
    """Resolve max_jobs: <=1 treated as fraction of CPU cores, >1 as absolute."""
    if value <= 1:
        return max(1, int(multiprocessing.cpu_count() * value))
    return int(value)


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         check: bool = True,
         ctx: BuildContext | None = None) -> subprocess.CompletedProcess:
    """Run a command, streaming each output line in real time."""
    if ctx is None:
        ctx = BuildContext()
    merged_env = None
    if env:
        merged_env = {**os.environ, **env}
    cmd_str = " ".join(cmd)
    print(f"{ctx.prefix}$ {cmd_str}", flush=True)
    if ctx.log_file:
        ctx.log_file.write(f"$ {cmd_str}\n")
    if ctx.prefix or ctx.log_file:
        # Stream output line-by-line for real-time progress
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        output_lines = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            print(f"{ctx.prefix}{line}", flush=True)
            if ctx.log_file:
                ctx.log_file.write(f"{line}\n")
        returncode = proc.wait()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        return subprocess.CompletedProcess(
            cmd, returncode, stdout="\n".join(output_lines), stderr=None)
    return subprocess.run(
        cmd, cwd=cwd, env=merged_env, check=check,
        stdout=sys.stdout, stderr=sys.stderr,
    )


def clone_or_update(repo_url: str, branch: str, commit: str,
                    repos_dir: Path,
                    ctx: BuildContext | None = None,
                    auto_git_pull: bool = True) -> Path:
    """Clone or fetch+checkout a branch into repos_dir/<sanitized-branch>/."""
    if ctx is None:
        ctx = BuildContext()
    dir_name = branch_to_dir(branch, commit)
    repo_dir = repos_dir / dir_name

    if not repo_dir.exists():
        ctx.log(f"Cloning {repo_url} -> {repo_dir}")
        _run(["git", "clone", repo_url, str(repo_dir)], ctx=ctx)
    elif not auto_git_pull:
        ctx.log(f"Skipping git pull (auto_git_pull=false), using existing {repo_dir}")
        return repo_dir
    else:
        # Ensure origin matches the configured repo URL
        _run(["git", "remote", "set-url", "origin", repo_url],
             cwd=repo_dir, ctx=ctx)

    ctx.log(f"Fetching {branch}...")
    _run(["git", "fetch", "origin", branch], cwd=repo_dir, ctx=ctx)

    if commit:
        _run(["git", "checkout", commit], cwd=repo_dir, ctx=ctx)
    else:
        _run(["git", "checkout", branch], cwd=repo_dir, ctx=ctx)
        _run(["git", "reset", "--hard", f"origin/{branch}"],
             cwd=repo_dir, ctx=ctx)

    return repo_dir


def setup_venv(repo_dir: Path, ctx: BuildContext | None = None) -> None:
    """Create per-repo venv if it doesn't exist."""
    if ctx is None:
        ctx = BuildContext()
    venv_dir = repo_dir / ".venv"
    if not venv_dir.exists():
        ctx.log(f"Creating venv in {repo_dir}...")
        _run(["uv", "venv", "--python", "3.12", "--seed"],
             cwd=repo_dir, ctx=ctx)


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
        "install_flash_attn": build.install_flash_attn,
        "torch_index": build.torch_index,
    }


def _build_identity(build: BuildConfig) -> dict:
    """Fields that affect the build artifact (excludes execution params)."""
    return {
        "use_precompiled": build.use_precompiled,
        "cuda_arch": build.cuda_arch,
        "install_flash_attn": build.install_flash_attn,
        "install_deepgemm": build.install_deepgemm,
        "install_flashinfer_jit_cache": build.install_flashinfer_jit_cache,
        "torch_index": build.torch_index,
    }


def _check_cuda_version(repo_dir: Path, ctx: BuildContext) -> None:
    """Warn/error when torch's CUDA version doesn't match the system.

    Only errors when torch CUDA is older than the system — this causes
    cuBLAS initialization failures on newer GPUs (e.g. cuBLAS 12 on
    SM 100+ B200). When torch CUDA is newer, it's a warning only since
    newer cuBLAS still supports older GPUs.
    """
    venv_python = str(repo_dir / ".venv" / "bin" / "python")
    try:
        torch_cuda = subprocess.check_output(
            [venv_python, "-c", "import torch; print(torch.version.cuda)"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        system_cuda = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # Can't check, skip

    if torch_cuda == "None":
        raise RuntimeError(
            "CPU-only torch installed on a CUDA system. "
            "Fix: install CUDA torch, e.g.:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cu130")

    m = re.search(r"release (\d+)\.", system_cuda)
    if not m:
        return
    system_major = int(m.group(1))
    torch_major = int(torch_cuda.split(".")[0])

    if torch_major < system_major:
        raise RuntimeError(
            f"CUDA version mismatch: torch has CUDA {torch_cuda} "
            f"but system has CUDA {system_major}.x. "
            f"Older cuBLAS may not support this GPU. "
            f"Fix: install torch with matching CUDA version, e.g.:\n"
            f"  pip install torch --index-url "
            f"https://download.pytorch.org/whl/cu{system_major}0")
    elif torch_major > system_major:
        ctx.log(f"NOTE: torch has CUDA {torch_cuda}, system has "
                f"CUDA {system_major}.x (OK for runtime, "
                f"torch bundles its own CUDA libraries)")


def build_vllm(repo_dir: Path, build: BuildConfig,
               max_jobs: int | None = None,
               ctx: BuildContext | None = None) -> None:
    """Install torch, deps, and build vllm in the repo's venv."""
    if ctx is None:
        ctx = BuildContext()
    current_state = _current_build_state(repo_dir, build)

    # Check if build can be skipped (FORCE_BUILD=1 bypasses cache)
    if os.environ.get("FORCE_BUILD") != "1":
        old_state = _read_build_state(repo_dir)
        if old_state == current_state:
            ctx.log(f"Build already done for "
                    f"{current_state['commit'][:12]} — skipping.")
            return

    venv_python = str(repo_dir / ".venv" / "bin" / "python")
    uv_pip = ["uv", "pip", "install", "--python", venv_python]
    jobs = _resolve_jobs(max_jobs or build.max_jobs)
    env: dict[str, str] = {"MAX_JOBS": str(jobs)}

    # Pass through build env vars
    for var in ("VLLM_FLASH_ATTN_SRC_DIR", "VLLM_CUTLASS_SRC_DIR",
                "VLLM_TARGET_DEVICE", "CMAKE_BUILD_TYPE"):
        if os.environ.get(var):
            env[var] = os.environ[var]

    # Use sccache/ccache for compiler caching if available
    _sccache = shutil.which("sccache")
    _ccache = shutil.which("ccache")
    if _sccache:
        cmake_args = env.get("CMAKE_ARGS", os.environ.get("CMAKE_ARGS", ""))
        cmake_args += (f" -DCMAKE_C_COMPILER_LAUNCHER={_sccache}"
                       f" -DCMAKE_CXX_COMPILER_LAUNCHER={_sccache}"
                       f" -DCMAKE_CUDA_COMPILER_LAUNCHER={_sccache}")
        env["CMAKE_ARGS"] = cmake_args
        ctx.log(f"Using sccache: {_sccache}")
    elif _ccache:
        cmake_args = env.get("CMAKE_ARGS", os.environ.get("CMAKE_ARGS", ""))
        cmake_args += (f" -DCMAKE_C_COMPILER_LAUNCHER={_ccache}"
                       f" -DCMAKE_CXX_COMPILER_LAUNCHER={_ccache}"
                       f" -DCMAKE_CUDA_COMPILER_LAUNCHER={_ccache}")
        env["CMAKE_ARGS"] = cmake_args
        ctx.log(f"Using ccache: {_ccache}")

    if build.use_precompiled:
        # Precompiled: single command, deps resolve from PyPI.
        # https://docs.vllm.ai/en/latest/contributing/#developing
        ctx.log(f"Installing vllm (precompiled, "
                f"HEAD={current_state['commit'][:12]})...")
        env["VLLM_USE_PRECOMPILED"] = "1"
        # uv needs the torch index to resolve CUDA wheels (pip doesn't).
        # --index-strategy unsafe-best-match: allow uv to pick best version
        # across all indexes (PyTorch index only has old cmake/ninja).
        _run(uv_pip + ["-e", ".", "--extra-index-url", build.torch_index,
                        "--index-strategy", "unsafe-best-match"],
             cwd=repo_dir, env=env, ctx=ctx)

    else:
        # Source build following vllm contributing docs:
        # https://docs.vllm.ai/en/latest/contributing/#developing
        ctx.log(f"Building vllm from source "
                f"(HEAD={current_state['commit'][:12]})...")

        # 1. Install torch + torchvision + torchaudio (CUDA)
        _run(uv_pip + ["torch", "torchvision", "torchaudio",
                        "--extra-index-url", build.torch_index], ctx=ctx)

        # 2. Install build deps (minus torch, already installed above)
        build_reqs = repo_dir / "requirements" / "build.txt"
        if build_reqs.exists():
            lines = []
            for raw in build_reqs.read_text().splitlines():
                line = raw.split("#")[0].strip()
                if line and not line.startswith("torch=="):
                    lines.append(line)
            if lines:
                _run(uv_pip + lines, ctx=ctx)

        # 3. Optionally install flash-attn
        if build.install_flash_attn:
            ctx.log("Installing flash-attn (+ build deps)...")
            _run(uv_pip + ["psutil", "packaging", "ninja"],
                 check=False, ctx=ctx)
            result = _run(uv_pip + ["flash-attn", "--no-build-isolation"],
                          check=False, ctx=ctx)
            if result.returncode != 0:
                ctx.log("WARNING: flash-attn install failed, skipping.")

        # 4. Build & install vllm
        if build.cuda_arch:
            env["TORCH_CUDA_ARCH_LIST"] = build.cuda_arch
            cmake_arch = build.cuda_arch.replace(".", "")
            cmake_args = env.get("CMAKE_ARGS",
                                 os.environ.get("CMAKE_ARGS", ""))
            env["CMAKE_ARGS"] = (
                f"{cmake_args} -DCMAKE_CUDA_ARCHITECTURES={cmake_arch}"
            )
        # Pass --extra-index-url so uv resolves torch from the CUDA
        # index (not PyPI CPU) when satisfying vllm's torch pin.
        _run(uv_pip + ["-e", ".", "--no-build-isolation",
                       "--extra-index-url", build.torch_index,
                       "--index-strategy", "unsafe-best-match"],
             cwd=repo_dir, env=env, ctx=ctx)

    # 5. Optionally install DeepGEMM (requires git clone with submodules)
    if build.install_deepgemm:
        import tempfile
        ctx.log("Installing DeepGEMM...")
        dg_dir = repo_dir / ".deepgemm"
        if not dg_dir.exists():
            _run(["git", "clone", "--recursive",
                  "https://github.com/deepseek-ai/DeepGEMM.git",
                  str(dg_dir)], ctx=ctx)
        result = _run(uv_pip + ["-e", str(dg_dir), "--no-build-isolation"],
                      check=False, ctx=ctx)
        if result.returncode != 0:
            ctx.log("WARNING: DeepGEMM install failed, skipping.")

    # 6. Optionally install flashinfer-jit-cache (pre-compiled JIT kernels)
    if build.install_flashinfer_jit_cache:
        import re as _re
        cuda_ver_match = _re.search(r"cu(\d+)", build.torch_index)
        cuda_ver = cuda_ver_match.group(1) if cuda_ver_match else "130"
        fi_index = f"https://flashinfer.ai/whl/cu{cuda_ver}"
        # Pin jit-cache version to match installed flashinfer-python
        fi_ver_result = _run(
            [venv_python, "-c",
             "import importlib.metadata; "
             "print(importlib.metadata.version('flashinfer-python'))"],
            check=False, ctx=ctx)
        fi_ver = fi_ver_result.stdout.strip() if fi_ver_result.returncode == 0 else ""
        pkg = f"flashinfer-jit-cache=={fi_ver}" if fi_ver else "flashinfer-jit-cache"
        ctx.log(f"Installing {pkg} (index: {fi_index})...")
        result = _run(
            uv_pip + [pkg, "--extra-index-url", fi_index],
            check=False, ctx=ctx)
        if result.returncode != 0:
            ctx.log("WARNING: flashinfer-jit-cache install failed, skipping.")

    # Verify torch CUDA version matches system CUDA major version
    _check_cuda_version(repo_dir, ctx)

    _write_build_state(repo_dir, current_state)
    ctx.log("Build complete.")


def _install_one(repo_url: str, build: BuildConfig,
                 branch: str, commit: str,
                 repos_dir: Path, max_jobs: int,
                 builder_id: int = 0,
                 logs_dir: Path | None = None) -> Path:
    """Install a single branch (clone + venv + build). Runs in subprocess."""
    prefix = f"[build {builder_id}] " if builder_id else ""
    log_fh = None
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        dir_name = branch_to_dir(branch, commit)
        log_fh = open(logs_dir / f"build-{dir_name}.log", "w")

    ctx = BuildContext(prefix=prefix, log_file=log_fh)
    try:
        ctx.log(f"{'=' * 44}")
        ctx.log(f"  Installing: {branch}{f' @ {commit}' if commit else ''}")
        ctx.log(f"{'=' * 44}")

        repo_dir = clone_or_update(repo_url, branch, commit,
                                   repos_dir, ctx=ctx,
                                   auto_git_pull=build.auto_git_pull)
        setup_venv(repo_dir, ctx=ctx)
        build_vllm(repo_dir, build, max_jobs=max_jobs, ctx=ctx)
        return repo_dir
    finally:
        if log_fh:
            log_fh.flush()
            log_fh.close()


def _unique_builds(config: Config) -> list[tuple[str, str, BuildConfig]]:
    """Unique (branch, commit, effective_build) triples, preserving order.

    Raises ValueError if the same (branch, commit) has conflicting
    per-run build overrides (comparing only artifact-affecting fields).
    """
    seen: dict[tuple[str, str], tuple[dict, BuildConfig]] = {}
    unique: list[tuple[str, str, BuildConfig]] = []
    for run in config.runs:
        key = (run.branch, run.commit)
        eff_build = config.effective_build(run)
        identity = _build_identity(eff_build)
        if key in seen:
            if seen[key][0] != identity:
                raise ValueError(
                    f"Conflicting build overrides for {run.branch}"
                    f"{f' @ {run.commit}' if run.commit else ''}: "
                    f"runs sharing a branch must have identical build configs")
        else:
            seen[key] = (identity, eff_build)
            unique.append((key[0], key[1], eff_build))
    return unique


def install_all(config: Config,
                repos_dir: Path,
                logs_dir: Path | None = None) -> dict[tuple[str, str], Path]:
    """Build all unique branches. Parallel when build_parallelism > 1.

    Returns mapping from (branch, commit) -> repo_dir.
    """
    unique = _unique_builds(config)
    max_jobs = _resolve_jobs(config.build.max_jobs)
    repo_url = config.project.repo
    parallelism = config.project.build_parallelism

    n = len(unique)
    print(f"Installing {n} unique branch(es)"
          f"{f' ({parallelism} parallel)' if parallelism > 1 else ''}...")

    results: dict[tuple[str, str], Path] = {}

    if parallelism <= 1 or n <= 1:
        for i, (branch, commit, build_cfg) in enumerate(unique):
            repo_dir = _install_one(repo_url, build_cfg, branch, commit,
                                    repos_dir, max_jobs=max_jobs,
                                    builder_id=i + 1 if n > 1 else 0,
                                    logs_dir=logs_dir)
            results[(branch, commit)] = repo_dir
    else:
        workers = min(parallelism, n)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, (branch, commit, build_cfg) in enumerate(unique):
                fut = pool.submit(
                    _install_one, repo_url, build_cfg, branch, commit,
                    repos_dir, max_jobs, builder_id=i + 1,
                    logs_dir=logs_dir)
                futures[fut] = (branch, commit)
            for fut in as_completed(futures):
                key = futures[fut]
                results[key] = fut.result()

    return results
