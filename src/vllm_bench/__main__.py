# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entry point for vllm-bench."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import Config, load_config
from .runner import (all_, bench, build, build_bench, build_eval, build_test,
                     compile, eval_, test, repos_dir_for)


def _print_banner(config: Config) -> None:
    proj = config.project
    bld = config.build
    srv = config.server
    bnch = config.bench

    print("=" * 44)
    print("  vLLM A/B Benchmark")
    print("=" * 44)
    print(f"Config:      {proj.name}")
    print(f"Model:       {proj.model}")
    print(f"TP:          {srv.tp}")
    print(f"Max len:     {srv.max_model_len}")
    print(f"Prompts:     {bnch.num_prompts}")
    print(f"Input len:   {bnch.input_len}")
    print(f"Output len:  {bnch.output_len}")
    print(f"Precompiled: {bld.use_precompiled}")
    print(f"Flash-attn:  {bld.install_flash_attn}")
    if bld.cuda_arch:
        print(f"CUDA arch:   {bld.cuda_arch}")
    if config.test.script:
        print(f"Test:        {config.test.script}")
    if config.eval.script:
        print(f"Eval:        {config.eval.script}")
    print(f"Work dir:    {proj.work_dir}")
    print(f"Runs:        {len(config.runs)}")
    for run in config.runs:
        eff_srv = config.effective_server(run)
        cc = ""
        if eff_srv.compilation_config:
            cc = f", cc={json.dumps(eff_srv.compilation_config)}"
        print(f"  - {run.label} (branch={run.branch}{cc})")
    print("=" * 44)
    print()


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="Path to config YAML")
    p.add_argument("--run", metavar="LABEL", action="append",
                   help="Only process this run label (repeatable)")


def _filter_runs(config: Config, labels: list[str] | None) -> Config:
    """Return a new Config with runs filtered to the given labels."""
    if not labels:
        return config
    label_set = set(labels)
    unknown = label_set - {r.label for r in config.runs}
    if unknown:
        print(f"ERROR: Unknown run label(s): {', '.join(sorted(unknown))}",
              file=sys.stderr)
        print(f"  Available: {', '.join(r.label for r in config.runs)}",
              file=sys.stderr)
        sys.exit(1)
    filtered = [r for r in config.runs if r.label in label_set]
    return Config(
        project=config.project,
        build=config.build,
        server=config.server,
        bench=config.bench,
        eval=config.eval,
        test=config.test,
        branches=config.branches,
        runs=filtered,
        config_path=config.config_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-bench",
        description="A/B benchmark for vLLM branches",
    )
    sub = parser.add_subparsers(dest="command")

    # -- build --
    p_build = sub.add_parser("build",
                             help="Clone and build all branches")
    _add_common_args(p_build)
    p_build.add_argument("--max-jobs", type=int, default=None)

    # -- compile --
    p_compile = sub.add_parser("compile",
                               help="Pre-compile CUDA graphs (start/check/stop)")
    _add_common_args(p_compile)
    p_compile.add_argument("--port", type=int, default=None)

    # -- build-bench --
    p_bbench = sub.add_parser("build-bench",
                              help="Build + benchmark")
    _add_common_args(p_bbench)
    p_bbench.add_argument("--port", type=int, default=None)
    p_bbench.add_argument("--max-jobs", type=int, default=None)

    # -- build-test --
    p_btest = sub.add_parser("build-test",
                             help="Build + test")
    _add_common_args(p_btest)
    p_btest.add_argument("--max-jobs", type=int, default=None)

    # -- build-eval --
    p_beval = sub.add_parser("build-eval",
                             help="Build + eval")
    _add_common_args(p_beval)
    p_beval.add_argument("--port", type=int, default=None)
    p_beval.add_argument("--max-jobs", type=int, default=None)

    # -- bench --
    p_bench = sub.add_parser("bench",
                             help="Benchmark only (builds must exist)")
    _add_common_args(p_bench)
    p_bench.add_argument("--port", type=int, default=None)

    # -- test --
    p_test = sub.add_parser("test",
                            help="Run tests only (builds must exist)")
    _add_common_args(p_test)

    # -- eval --
    p_eval = sub.add_parser("eval",
                            help="Eval only (builds must exist)")
    _add_common_args(p_eval)
    p_eval.add_argument("--port", type=int, default=None)

    # -- all --
    p_all = sub.add_parser("all",
                           help="Build + test + eval + benchmark")
    _add_common_args(p_all)
    p_all.add_argument("--port", type=int, default=None)
    p_all.add_argument("--max-jobs", type=int, default=None)

    # -- clean --
    p_clean = sub.add_parser("clean",
                             help="Remove caches that can cause stale builds")
    p_clean.add_argument("config", nargs="?", default=None,
                         help="Config YAML (also cleans venvs for its runs)")
    p_clean.add_argument("--all", action="store_true",
                         help="Remove everything including model cache")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "clean":
        _clean(args)
        return

    config = load_config(
        args.config,
        port=getattr(args, "port", None),
        max_jobs=getattr(args, "max_jobs", None),
    )
    config = _filter_runs(config, args.run)

    _print_banner(config)

    if args.command == "build":
        build(config)
    elif args.command == "compile":
        compile(config)
    elif args.command == "build-bench":
        build_bench(config)
    elif args.command == "build-test":
        build_test(config)
    elif args.command == "build-eval":
        build_eval(config)
    elif args.command == "bench":
        bench(config)
    elif args.command == "test":
        test(config)
    elif args.command == "eval":
        eval_(config)
    elif args.command == "all":
        all_(config)


# ── clean ────────────────────────────────────────────────────────────

_SYSTEM_CACHES = [
    ("flashinfer JIT", Path.home() / ".cache" / "flashinfer"),
    ("torch extensions", Path.home() / ".cache" / "torch_extensions"),
    ("vllm compilation", Path.home() / ".cache" / "vllm"),
    ("triton", Path.home() / ".triton" / "cache"),
]

_MODEL_CACHES = [
    ("huggingface models", Path.home() / ".cache" / "huggingface"),
]


def _clean(args: argparse.Namespace) -> None:
    """Remove caches that can cause stale/broken builds."""
    removed = []

    # System caches (always)
    caches = list(_SYSTEM_CACHES)
    if args.all:
        caches += _MODEL_CACHES

    for label, path in caches:
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*")
                       if f.is_file()) / (1024 * 1024)
            shutil.rmtree(path)
            removed.append(f"  {label}: {path} ({size:.0f} MB)")

    # Config-specific: clean venvs and build state
    if args.config:
        config = load_config(args.config)
        repos_dir = repos_dir_for(config)
        from .builder import branch_to_dir
        for run in config.runs:
            d = repos_dir / branch_to_dir(run.branch, run.commit)
            venv = d / ".venv"
            state = d / ".build_state.json"
            if venv.exists():
                shutil.rmtree(venv)
                removed.append(f"  venv: {venv}")
            if state.exists():
                state.unlink()
                removed.append(f"  build state: {state}")

    if removed:
        print("Cleaned:")
        print("\n".join(removed))
    else:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
