# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestration: parallel builds -> sequential benchmark runs."""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .builder import branch_to_dir, install_all
from .config import Config
from .resolved import ResolvedRun, resolve_runs
from .server import Server
from .summary import format_eval_summary, format_summary, format_test_summary


def _repo_owner_name(repo_url: str) -> str:
    """Extract owner/name from git URL."""
    m = re.search(r"[/:]([^/:]+)/([^/]+?)(?:\.git)?$", repo_url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "unknown/repo"


def repos_dir_for(config: Config) -> Path:
    """Deterministic repos directory from config."""
    work_dir = Path(config.project.work_dir)
    owner_name = _repo_owner_name(config.project.repo)
    return work_dir / "repos" / owner_name


def _require_builds(config: Config,
                    resolved: list[ResolvedRun] | None = None,
                    ) -> list[ResolvedRun]:
    """Validate that builds exist for runs. Returns resolved runs."""
    if resolved is None:
        repos_dir = repos_dir_for(config)
        resolved = resolve_runs(config, repos_dir)
    for r in resolved:
        d = r.repo_dir
        if not (d / ".venv").exists():
            print(f"ERROR: No build found for {r.branch}"
                  f"{f' @ {r.commit}' if r.commit else ''}\n"
                  f"  Expected: {d}\n"
                  f"  Run 'vllm-bench build' first.",
                  file=sys.stderr)
            sys.exit(1)
        if not (d / ".build_state.json").exists():
            print(f"WARNING: Build state missing for {r.branch}"
                  f"{f' @ {r.commit}' if r.commit else ''}\n"
                  f"  Venv exists but no .build_state.json ({d})\n"
                  f"  Build caching will not work for this branch.",
                  file=sys.stderr)
    return resolved


def _kill_gpu_processes() -> None:
    """Kill any leftover GPU compute processes (e.g. vLLM EngineCore)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().splitlines():
            pid = line.strip()
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _logs_dir(config: Config, phase: str, timestamp: str) -> Path:
    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")
    return work_dir / "logs" / name / f"{phase}-{timestamp}"


def _symlink_current(directory: Path) -> None:
    """Create a 'current' symlink under logs/ or results/ pointing to directory.

    Given e.g. results/<config_name>/eval-20260405/, creates
    results/current -> <config_name>/eval-20260405/
    """
    # directory.parent = results/<config_name>, grandparent = results/
    top = directory.parent.parent
    link = top / "current"
    rel = directory.relative_to(top)
    link.unlink(missing_ok=True)
    link.symlink_to(rel)


def build(config: Config, timestamp: str | None = None) -> Path:
    """Clone and build all unique branches. Returns repos_dir."""
    repos_dir = repos_dir_for(config)
    repos_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = _logs_dir(config, "build", timestamp or _make_timestamp())

    install_all(config, repos_dir, logs_dir=logs_dir)
    print(f"Build logs in: {logs_dir}/")
    return repos_dir


def _compile_one(resolved: ResolvedRun, config: Config,
                 logs_dir: Path, prefix: str = "",
                 flush_jitter: float = 0.0) -> str:
    """Start server, compile CUDA graphs, sanity check, stop. Returns label."""
    with Server(resolved, config, logs_dir,
                prefix=prefix, flush_jitter=flush_jitter):
        if prefix:
            print(f"{prefix}Server for {resolved.label!r} compiled "
                  f"successfully.", flush=True)
        else:
            print(f"Server for {resolved.label!r} compiled successfully.")
    return resolved.label


def compile(config: Config, timestamp: str | None = None) -> None:
    """Start, compile CUDA graphs, sanity check, and stop each server.

    Parallel when compile_parallelism > 1 (each server gets a unique port).
    """
    resolved_runs = _require_builds(config)
    logs_dir = _logs_dir(config, "compile", timestamp or _make_timestamp())
    logs_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(resolved_runs)
    parallelism = config.project.compile_parallelism

    if parallelism <= 1 or n_runs <= 1:
        print(f"Compiling {n_runs} run(s)...")
        for i, resolved in enumerate(resolved_runs):
            prefix = f"[compile{i + 1}] " if n_runs > 1 else ""
            _compile_one(resolved=resolved, config=config, logs_dir=logs_dir,
                         prefix=prefix)
    else:
        workers = min(parallelism, n_runs)
        print(f"Compiling {n_runs} run(s) ({workers} parallel)...")
        base_port = config.server.port
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, resolved in enumerate(resolved_runs):
                r = resolved.with_server(port=base_port + i)
                jitter = random.uniform(0, 0.3) * i
                fut = pool.submit(
                    _compile_one, resolved=r, config=config,
                    logs_dir=logs_dir,
                    prefix=f"[compile{i + 1}] ",
                    flush_jitter=jitter)
                futures[fut] = resolved.label
            for fut in as_completed(futures):
                fut.result()

    print("All servers compiled and verified.")


def _setup_run_dirs(config: Config,
                    timestamp: str | None = None) -> tuple[Path, Path]:
    """Create results and logs directories for a benchmark run."""
    ts = timestamp or _make_timestamp()
    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")

    results_dir = work_dir / "results" / name / f"bench-{ts}"
    logs_dir = _logs_dir(config, "bench", ts)

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _symlink_current(results_dir)
    _symlink_current(logs_dir)

    return results_dir, logs_dir


def _execute_benchmark(resolved: ResolvedRun, config: Config,
                       results_dir: Path,
                       prefix: str = "",
                       suffix: str = "") -> Path:
    """Run vllm bench serve and capture output."""
    bench = resolved.bench
    srv = resolved.server
    stem = f"{resolved.label}{suffix}"
    outfile = results_dir / f"{stem}.txt"
    vllm_bin = str(resolved.vllm_bin)

    cmd = [
        vllm_bin, "bench", "serve",
        "--backend", "vllm",
        "--model", config.project.model,
        "--port", str(srv.port),
        "--num-prompts", str(bench.num_prompts),
        "--request-rate", str(bench.request_rate),
        "--random-input-len", str(bench.input_len),
        "--random-output-len", str(bench.output_len),
        "--num-warmups", str(bench.num_warmups),
        "--ignore-eos",
    ]
    if bench.max_concurrency is not None:
        cmd += ["--max-concurrency", str(bench.max_concurrency)]

    print(f"{prefix}$ {' '.join(cmd)}", flush=True)

    with open(outfile, "w") as out_f:
        proc = subprocess.Popen(
            cmd,
            cwd=resolved.repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            out_f.write(line)
            if '"POST /v1/completions HTTP/1.1" 200 OK' not in line:
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"Benchmark for {resolved.label!r} failed (exit code {rc}). "
            f"Output saved to {outfile}")

    print(f"{prefix}Results saved to {outfile}", flush=True)
    return outfile


def bench(config: Config, timestamp: str | None = None) -> dict[str, Path]:
    """Run benchmarks assuming builds already exist.

    Returns mapping of label -> result file path.
    """
    resolved_runs = _require_builds(config)
    results_dir, logs_dir = _setup_run_dirs(config, timestamp)

    # Save config for reproducibility
    if config.config_path and config.config_path.exists():
        shutil.copy2(config.config_path, results_dir / "config.yaml")

    print(f"Running {len(resolved_runs)} benchmark(s)...")
    results: dict[str, Path] = {}
    for i, resolved in enumerate(resolved_runs):
        iters = resolved.bench.iterations
        prefix = f"[bench {i + 1}] "
        with Server(resolved, config, logs_dir,
                     prefix=prefix) as server:
            server.warmup(resolved.bench.warmup_prompts)
            for it in range(iters):
                suffix = f"_iter{it + 1}" if iters > 1 else ""
                it_prefix = (f"[bench {i + 1} iter {it + 1}/{iters}] "
                             if iters > 1 else prefix)
                result_path = _execute_benchmark(
                    resolved, config, results_dir,
                    prefix=it_prefix, suffix=suffix)
            # Use last iteration for summary
            results[resolved.label] = result_path

    # Generate summary
    repos_dir = repos_dir_for(config)
    summary = format_summary(config, results, resolved_runs, repos_dir)
    summary_file = results_dir / "summary.txt"
    summary_file.write_text(summary)
    print(summary)
    print(f"Full results in: {results_dir}/")
    print(f"Server logs in:  {logs_dir}/")

    return results


def build_bench(config: Config) -> dict[str, Path]:
    """Build all branches then run benchmarks."""
    ts = _make_timestamp()
    build(config, timestamp=ts)
    if config.project.compile_parallelism > 1:
        compile(config, timestamp=ts)
    return bench(config, timestamp=ts)


# ── Eval ──────────────────────────────────────────────────────────────


def _execute_eval(resolved: ResolvedRun, config: Config,
                  results_dir: Path,
                  prefix: str = "") -> Path:
    """Run an eval script against a live server and capture results."""
    eval_cfg = resolved.eval
    srv = resolved.server
    outfile = results_dir / f"{resolved.label}.json"

    cmd = [
        str(resolved.venv_python),
        str(resolved.repo_dir / eval_cfg.script),
        *shlex.split(eval_cfg.args),
        "--host", "http://127.0.0.1",
        "--port", str(srv.port),
        "--save-results", str(outfile),
    ]

    print(f"{prefix}$ {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=resolved.repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        sys.stdout.write(f"{prefix}{line}")
        sys.stdout.flush()
    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"Eval for {resolved.label!r} failed (exit code {rc}).")

    print(f"{prefix}Results saved to {outfile}", flush=True)
    return outfile


def _setup_eval_dirs(config: Config,
                     timestamp: str | None = None) -> tuple[Path, Path]:
    """Create results and logs directories for an eval run."""
    ts = timestamp or _make_timestamp()
    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")

    results_dir = work_dir / "results" / name / f"eval-{ts}"
    logs_dir = _logs_dir(config, "eval", ts)

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _symlink_current(results_dir)
    _symlink_current(logs_dir)

    return results_dir, logs_dir


def eval_(config: Config, timestamp: str | None = None) -> dict[str, Path]:
    """Run evals assuming builds already exist.

    Returns mapping of label -> result file path.
    """
    repos_dir = repos_dir_for(config)
    all_runs = resolve_runs(config, repos_dir)
    eval_runs = [r for r in all_runs if r.eval.script]

    if not eval_runs:
        print("No runs have eval.script configured, skipping eval.")
        return {}

    _require_builds(config, resolved=eval_runs)
    results_dir, logs_dir = _setup_eval_dirs(config, timestamp)

    # Save config for reproducibility
    if config.config_path and config.config_path.exists():
        shutil.copy2(config.config_path, results_dir / "config.yaml")

    print(f"Running {len(eval_runs)} eval(s)...")
    results: dict[str, Path] = {}
    for i, resolved in enumerate(eval_runs):
        prefix = f"[eval {i + 1}] "
        with Server(resolved, config, logs_dir,
                    prefix=prefix) as server:
            server.warmup(resolved.bench.warmup_prompts)
            result_path = _execute_eval(
                resolved, config, results_dir, prefix=prefix)
            results[resolved.label] = result_path

    # Generate eval summary
    summary = format_eval_summary(config, results, eval_runs)
    summary_file = results_dir / "summary.txt"
    summary_file.write_text(summary)
    print(summary)
    print(f"Eval results in: {results_dir}/")
    print(f"Server logs in:  {logs_dir}/")

    return results


def build_eval(config: Config) -> dict[str, Path]:
    """Build all branches then run evals."""
    ts = _make_timestamp()
    build(config, timestamp=ts)
    return eval_(config, timestamp=ts)


# ── Test ─────────────────────────────────────────────────────────────


def _setup_test_dirs(config: Config,
                     timestamp: str | None = None) -> tuple[Path, Path]:
    """Create results and logs directories for a test run."""
    ts = timestamp or _make_timestamp()
    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")

    results_dir = work_dir / "results" / name / f"test-{ts}"
    logs_dir = _logs_dir(config, "test", ts)

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _symlink_current(results_dir)
    _symlink_current(logs_dir)

    return results_dir, logs_dir


_TEST_DEPS = ["pytest", "tblib"]


def _ensure_pytest(resolved: ResolvedRun, prefix: str = "") -> None:
    """Install pytest and test deps into the run's venv if not present."""
    missing = []
    for dep in _TEST_DEPS:
        rc = subprocess.run(
            [str(resolved.venv_python), "-c", f"import {dep}"],
            capture_output=True).returncode
        if rc != 0:
            missing.append(dep)
    if not missing:
        return
    print(f"{prefix}Installing test dependencies: {', '.join(missing)}",
          flush=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(resolved.venv_python)]
        + missing,
        check=True)


def _execute_test(resolved: ResolvedRun, config: Config,
                  results_dir: Path,
                  prefix: str = "") -> tuple[Path, int]:
    """Run pytest in the repo venv and capture output.

    Returns (output_file, return_code). Does NOT raise on test failure
    so that all runs can complete before reporting.
    """
    _ensure_pytest(resolved, prefix=prefix)

    test_cfg = resolved.test
    outfile = results_dir / f"{resolved.label}.txt"

    cmd = [
        str(resolved.venv_python), "-m", "pytest",
        str(resolved.repo_dir / test_cfg.script),
        *shlex.split(test_cfg.args),
    ]

    print(f"{prefix}$ {' '.join(cmd)}", flush=True)

    with open(outfile, "w") as out_f:
        proc = subprocess.Popen(
            cmd,
            cwd=resolved.repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        for line in proc.stdout:
            out_f.write(line)
            out_f.flush()
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()
        rc = proc.wait()
        # Kill any leftover child processes (vLLM EngineCore spawned via
        # multiprocessing.spawn creates new process groups, so killpg
        # won't reach them). Use psutil-style cleanup.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Also kill any orphaned vLLM engine processes on our GPUs
        _kill_gpu_processes()
        time.sleep(2)  # allow GPU memory to be reclaimed

    status = "PASSED" if rc == 0 else "FAILED"
    print(f"{prefix}Tests {status} (exit code {rc}). "
          f"Output saved to {outfile}", flush=True)
    return outfile, rc


def test(config: Config, timestamp: str | None = None) -> dict[str, Path]:
    """Run tests assuming builds already exist.

    Returns mapping of label -> result file path.
    """
    repos_dir = repos_dir_for(config)
    all_runs = resolve_runs(config, repos_dir)
    test_runs = [r for r in all_runs if r.test.script]

    if not test_runs:
        print("No runs have test.script configured, skipping test.")
        return {}

    _require_builds(config, resolved=test_runs)
    results_dir, logs_dir = _setup_test_dirs(config, timestamp)

    # Save config for reproducibility
    if config.config_path and config.config_path.exists():
        shutil.copy2(config.config_path, results_dir / "config.yaml")

    print(f"Running {len(test_runs)} test suite(s)...")
    results: dict[str, Path] = {}
    failures: list[str] = []
    for i, resolved in enumerate(test_runs):
        prefix = f"[test {i + 1}] "
        result_path, rc = _execute_test(
            resolved, config, results_dir, prefix=prefix)
        results[resolved.label] = result_path
        if rc != 0:
            failures.append(resolved.label)

    # Generate test summary
    summary = format_test_summary(config, results, test_runs, failures)
    summary_file = results_dir / "summary.txt"
    summary_file.write_text(summary)
    print(summary)
    print(f"Test results in: {results_dir}/")

    if failures:
        print(f"FAILED runs: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)

    return results


def build_test(config: Config) -> dict[str, Path]:
    """Build all branches then run tests."""
    ts = _make_timestamp()
    build(config, timestamp=ts)
    return test(config, timestamp=ts)


def all_(config: Config) -> dict[str, Path]:
    """Build all branches, run tests/evals (if configured), then benchmark."""
    ts = _make_timestamp()
    build(config, timestamp=ts)
    if config.project.compile_parallelism > 1:
        compile(config, timestamp=ts)
    test(config, timestamp=ts)
    eval_(config, timestamp=ts)
    return bench(config, timestamp=ts)
