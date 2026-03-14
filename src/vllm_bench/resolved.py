# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-computed run configuration with resolved paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .config import (BenchConfig, BuildConfig, Config, EvalConfig, RunConfig,
                     ServerConfig)


@dataclass
class ResolvedRun:
    """A single benchmark run with all configs and paths pre-computed."""

    run: RunConfig
    build: BuildConfig
    server: ServerConfig
    bench: BenchConfig
    eval: EvalConfig
    repo_dir: Path
    venv_python: Path
    vllm_bin: Path

    @property
    def label(self) -> str:
        return self.run.label

    @property
    def branch(self) -> str:
        return self.run.branch

    @property
    def commit(self) -> str:
        return self.run.commit

    def with_server(self, **overrides) -> ResolvedRun:
        """Return a copy with server config fields overridden."""
        return replace(self, server=replace(self.server, **overrides))


def resolve_runs(config: Config, repos_dir: Path) -> list[ResolvedRun]:
    """Resolve all runs in a config into ResolvedRun instances."""
    from .builder import branch_to_dir

    resolved = []
    for run in config.runs:
        repo_dir = repos_dir / branch_to_dir(run.branch, run.commit)
        resolved.append(ResolvedRun(
            run=run,
            build=config.effective_build(run),
            server=config.effective_server(run),
            bench=config.effective_bench(run),
            eval=config.effective_eval(run),
            repo_dir=repo_dir,
            venv_python=repo_dir / ".venv" / "bin" / "python3",
            vllm_bin=repo_dir / ".venv" / "bin" / "vllm",
        ))
    return resolved
