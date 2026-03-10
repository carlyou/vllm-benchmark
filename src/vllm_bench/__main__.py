# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI entry point for vllm-bench."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config, load_config
from .runner import bench, build, run_all, serve


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
        runs=filtered,
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

    # -- serve --
    p_serve = sub.add_parser("serve",
                              help="Start/warmup/stop servers (no benchmark)")
    _add_common_args(p_serve)
    p_serve.add_argument("--port", type=int, default=None)

    # -- run --
    p_run = sub.add_parser("run",
                           help="Run benchmarks (builds must exist)")
    _add_common_args(p_run)
    p_run.add_argument("--port", type=int, default=None)

    # -- all --
    p_all = sub.add_parser("all",
                           help="Build + serve + run (default)")
    _add_common_args(p_all)
    p_all.add_argument("--port", type=int, default=None)
    p_all.add_argument("--max-jobs", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
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
    elif args.command == "serve":
        serve(config)
    elif args.command == "run":
        bench(config)
    else:
        run_all(config)


if __name__ == "__main__":
    main()
