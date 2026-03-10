# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestration: parallel builds -> sequential benchmark runs."""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from .builder import branch_to_dir, install_all
from .config import Config, RunConfig
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


def _require_builds(config: Config) -> Path:
    """Validate that builds exist for all runs. Returns repos_dir."""
    repos_dir = repos_dir_for(config)
    for run in config.runs:
        d = repos_dir / branch_to_dir(run.branch, run.commit)
        if not (d / ".venv").exists():
            print(f"ERROR: No build found for {run.branch}"
                  f"{f' @ {run.commit}' if run.commit else ''}\n"
                  f"  Expected: {d}\n"
                  f"  Run 'vllm-bench build' first.",
                  file=sys.stderr)
            sys.exit(1)
    return repos_dir


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


def _compile_one(repo_dir: Path, config: Config, run: RunConfig,
                 logs_dir: Path, port: int,
                 prefix: str = "", flush_jitter: float = 0.0) -> str:
    """Start server, compile CUDA graphs, sanity check, stop. Returns label."""
    compile_run = RunConfig(
        label=run.label, branch=run.branch, commit=run.commit,
        build=run.build,
        server={**run.server, "port": port},
        bench=run.bench,
    )
    with Server(repo_dir, config, compile_run, logs_dir,
                prefix=prefix, flush_jitter=flush_jitter):
        if prefix:
            print(f"{prefix}Server for {run.label!r} compiled successfully.",
                  flush=True)
        else:
            print(f"Server for {run.label!r} compiled successfully.")
    return run.label


def serve(config: Config) -> None:
    """Start, compile CUDA graphs, sanity check, and stop servers.

    When server.parallel_compile is True, all servers compile concurrently
    on auto-assigned ports (base_port + offset per run).
    """
    repos_dir = _require_builds(config)
    _, logs_dir = _setup_run_dirs(config)
    base_port = config.server.port

    workers = config.server.parallel_compile
    n_runs = len(config.runs)

    if workers > 1 and n_runs > 1:
        workers = min(workers, n_runs)
        print(f"Compiling {n_runs} run(s) ({workers} parallel)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for i, run in enumerate(config.runs):
                repo_dir = repos_dir / branch_to_dir(run.branch, run.commit)
                port = base_port + i
                prefix = f"[serve {i + 1}] "
                futures[pool.submit(
                    _compile_one, repo_dir, config, run, logs_dir, port,
                    prefix=prefix, flush_jitter=i * 1.0,
                )] = run.label
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
        for i, run in enumerate(config.runs):
            repo_dir = repos_dir / branch_to_dir(run.branch, run.commit)
            prefix = f"[serve {i + 1}] "
            _compile_one(repo_dir, config, run, logs_dir, base_port + i,
                         prefix=prefix)

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


def _execute_benchmark(repo_dir: Path, config: Config,
                       run: RunConfig, results_dir: Path,
                       prefix: str = "") -> Path:
    """Run vllm bench serve and capture output."""
    bench = config.effective_bench(run)
    srv = config.effective_server(run)
    outfile = results_dir / f"{run.label}.txt"
    venv_activate = repo_dir / ".venv" / "bin" / "activate"

    cmd = (
        f"source {venv_activate} && "
        f"vllm bench serve "
        f"--backend vllm "
        f"--model {config.project.model} "
        f"--port {srv.port} "
        f"--num-prompts {bench.num_prompts} "
        f"--request-rate inf "
        f"--random-input-len {bench.input_len} "
        f"--random-output-len {bench.output_len} "
        f"--ignore-eos"
    )

    print(f"{prefix}$ vllm bench serve --model {config.project.model} "
          f"--port {srv.port} --num-prompts {bench.num_prompts} "
          f"--random-input-len {bench.input_len} "
          f"--random-output-len {bench.output_len}",
          flush=True)

    with open(outfile, "w") as out_f:
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            out_f.write(line)
            if '"POST /v1/completions HTTP/1.1" 200 OK' not in line:
                sys.stdout.write(f"{prefix}{line}")
                sys.stdout.flush()
        proc.wait()

    print(f"{prefix}Results saved to {outfile}", flush=True)
    return outfile


def bench(config: Config) -> dict[str, Path]:
    """Run benchmarks assuming builds already exist.

    Returns mapping of label -> result file path.
    """
    repos_dir = _require_builds(config)
    results_dir, logs_dir = _setup_run_dirs(config)

    print(f"Running {len(config.runs)} benchmark(s)...")
    results: dict[str, Path] = {}
    for i, run in enumerate(config.runs):
        repo_dir = repos_dir / branch_to_dir(run.branch, run.commit)
        bench_cfg = config.effective_bench(run)
        prefix = f"[bench {i + 1}] "
        with Server(repo_dir, config, run, logs_dir,
                     prefix=prefix) as server:
            server.warmup(bench_cfg.warmup_prompts)
            result_path = _execute_benchmark(
                repo_dir, config, run, results_dir, prefix=prefix)
            results[run.label] = result_path

    # Generate summary
    summary = format_summary(config, results, repos_dir)
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
