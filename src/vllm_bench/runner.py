# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestration: parallel builds -> sequential benchmark runs."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .builder import branch_to_dir, install_all
from .config import Config
from .resolved import ResolvedRun, resolve_runs
from .server import Server
from .summary import format_summary


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


def _require_builds(config: Config) -> list[ResolvedRun]:
    """Validate that builds exist for all runs. Returns resolved runs."""
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


def build(config: Config) -> Path:
    """Clone and build all unique branches. Returns repos_dir."""
    repos_dir = repos_dir_for(config)
    repos_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = work_dir / "logs" / f"{name}-build-{timestamp}"

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


def serve(config: Config) -> None:
    """Start, compile CUDA graphs, sanity check, and stop servers.

    When server.parallel_compile > 1, all servers compile concurrently
    on auto-assigned ports (base_port + offset per run).
    """
    resolved_runs = _require_builds(config)
    work_dir = Path(config.project.work_dir)
    name = config.project.name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = work_dir / "logs" / f"{name}-serve-{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    base_port = config.server.port

    workers = config.server.parallel_compile
    n_runs = len(resolved_runs)

    if workers > 1 and n_runs > 1:
        workers = min(workers, n_runs)
        print(f"Compiling {n_runs} run(s) ({workers} parallel)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, resolved in enumerate(resolved_runs):
                port = base_port + i
                prefix = f"[serve {i + 1}] "
                r = resolved.with_server(port=port)
                futures[pool.submit(
                    _compile_one, r, config, logs_dir,
                    prefix=prefix, flush_jitter=i * 1.0,
                )] = resolved.label
            failed = []
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((label, e))
                    print(f"ERROR: Server for {label!r} failed: {e}",
                          file=sys.stderr)
            if failed:
                labels = ", ".join(f"'{l}'" for l, _ in failed)
                raise RuntimeError(
                    f"{len(failed)} server(s) failed to compile: {labels}. "
                    f"Try reducing server.parallel_compile")
    else:
        print(f"Compiling {n_runs} run(s)...")
        for i, resolved in enumerate(resolved_runs):
            port = base_port + i
            prefix = f"[serve {i + 1}] "
            r = resolved.with_server(port=port)
            _compile_one(r, config, logs_dir, prefix=prefix)

    print("All servers compiled and verified.")


def _setup_run_dirs(config: Config) -> tuple[Path, Path]:
    """Create results and logs directories for a benchmark run."""
    work_dir = Path(config.project.work_dir)

    # Use project name (e.g. "mla_quant_fusion/h100_fp8") sanitized for path
    name = config.project.name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{name}-{timestamp}"

    results_dir = work_dir / "results" / run_id
    logs_dir = work_dir / "logs" / run_id

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, logs_dir


def _execute_benchmark(resolved: ResolvedRun, config: Config,
                       results_dir: Path,
                       prefix: str = "") -> Path:
    """Run vllm bench serve and capture output."""
    bench = resolved.bench
    srv = resolved.server
    outfile = results_dir / f"{resolved.label}.txt"
    vllm_bin = str(resolved.vllm_bin)

    cmd = [
        vllm_bin, "bench", "serve",
        "--backend", "vllm",
        "--model", config.project.model,
        "--port", str(srv.port),
        "--num-prompts", str(bench.num_prompts),
        "--request-rate", "inf",
        "--random-input-len", str(bench.input_len),
        "--random-output-len", str(bench.output_len),
        "--ignore-eos",
    ]

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


def bench(config: Config) -> dict[str, Path]:
    """Run benchmarks assuming builds already exist.

    Returns mapping of label -> result file path.
    """
    resolved_runs = _require_builds(config)
    results_dir, logs_dir = _setup_run_dirs(config)

    # Save config for reproducibility
    if config.config_path and config.config_path.exists():
        shutil.copy2(config.config_path, results_dir / "config.yaml")

    print(f"Running {len(resolved_runs)} benchmark(s)...")
    results: dict[str, Path] = {}
    for i, resolved in enumerate(resolved_runs):
        prefix = f"[bench {i + 1}] "
        with Server(resolved, config, logs_dir,
                     prefix=prefix) as server:
            server.warmup(resolved.bench.warmup_prompts)
            result_path = _execute_benchmark(
                resolved, config, results_dir, prefix=prefix)
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


def run_all(config: Config) -> dict[str, Path]:
    """Build all branches, optionally pre-compile servers, then benchmark."""
    build(config)
    if config.server.parallel_compile > 1:
        serve(config)
    return bench(config)
