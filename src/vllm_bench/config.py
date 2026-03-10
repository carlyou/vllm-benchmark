# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config loading with structured sections and defaults.yaml inheritance."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path

import yaml


@dataclass
class ProjectConfig:
    repo: str = "https://github.com/vllm-project/vllm.git"
    model: str = ""
    name: str = ""  # derived from config path: "project_dir/config_stem"
    description: str = ""
    work_dir: str = "/tmp/vllm-bench"


@dataclass
class BuildConfig:
    use_precompiled: bool = True
    prebuilt_flash_attn: bool = True
    cuda_arch: str | None = None
    max_jobs: int = 16
    torch_index: str = "https://download.pytorch.org/whl/cu130"


@dataclass
class ServerConfig:
    tp: int = 1
    max_model_len: int = 4096
    enforce_eager: bool = False
    gpu_memory_utilization: float | None = None
    port: int = 8000
    wait_timeout: int = 600
    compilation_config: dict | None = None
    parallel_compile: int = 1


@dataclass
class BenchConfig:
    num_prompts: int = 200
    input_len: int = 128
    output_len: int = 128
    warmup_prompts: int = 3


@dataclass
class RunConfig:
    label: str
    branch: str
    commit: str = ""
    # Per-run overrides (merged with top-level via Config.effective_*)
    build: dict = field(default_factory=dict)
    server: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)


def _overlay(base, overrides: dict, cls):
    """Create a new dataclass instance by overlaying overrides on base."""
    if not overrides:
        return base
    kwargs = {f.name: getattr(base, f.name) for f in dc_fields(base)}
    kwargs.update(overrides)
    return cls(**kwargs)


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    runs: list[RunConfig] = field(default_factory=list)

    def effective_build(self, run: RunConfig) -> BuildConfig:
        """Top-level build config with per-run overrides applied."""
        return _overlay(self.build, run.build, BuildConfig)

    def effective_server(self, run: RunConfig) -> ServerConfig:
        """Top-level server config with per-run overrides applied."""
        return _overlay(self.server, run.server, ServerConfig)

    def effective_bench(self, run: RunConfig) -> BenchConfig:
        """Top-level bench config with per-run overrides applied."""
        return _overlay(self.bench, run.bench, BenchConfig)


# ── YAML loading ─────────────────────────────────────────────────────

_SECTIONS = {"project", "build", "server", "bench", "runs"}

_SECTION_FIELDS = {
    "project": {f.name for f in dc_fields(ProjectConfig)},
    "build": {f.name for f in dc_fields(BuildConfig)},
    "server": {f.name for f in dc_fields(ServerConfig)},
    "bench": {f.name for f in dc_fields(BenchConfig)},
}


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}



def _warn_unknown_fields(section: str, raw: dict) -> None:
    known = _SECTION_FIELDS.get(section, set())
    for key in raw:
        if key not in known:
            warnings.warn(
                f"Unknown field in {section!r}: {key!r}", stacklevel=3)


def _build_section(cls, raw: dict, section_name: str):
    """Build a dataclass from raw dict, warning on unknown fields."""
    _warn_unknown_fields(section_name, raw)
    known = {f.name for f in dc_fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return cls(**filtered)


def _parse_runs(raw_runs: list[dict]) -> list[RunConfig]:
    runs = []
    for r in raw_runs:
        runs.append(RunConfig(
            label=r["label"],
            branch=r["branch"],
            commit=r.get("commit", ""),
            build=r.get("build") or {},
            server=r.get("server") or {},
            bench=r.get("bench") or {},
        ))
    return runs


def load_config(config_path: str, **overrides) -> Config:
    """Load a benchmark config YAML.

    Optional CLI overrides: port, max_jobs.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = _load_yaml(path)

    # Warn on unknown top-level keys
    for key in raw:
        if key not in _SECTIONS:
            warnings.warn(f"Unknown top-level config key: {key!r}")

    # Build section configs
    raw_project = raw.get("project") or {}
    if "work_dir" in raw_project:
        raw_project["work_dir"] = os.path.expanduser(str(raw_project["work_dir"]))

    project = _build_section(ProjectConfig, raw_project, "project")
    build = _build_section(BuildConfig, raw.get("build") or {}, "build")
    server = _build_section(ServerConfig, raw.get("server") or {}, "server")
    bench = _build_section(BenchConfig, raw.get("bench") or {}, "bench")
    runs = _parse_runs(raw.get("runs") or [])

    # CLI overrides (machine-local conveniences only)
    if overrides.get("port") is not None:
        server = _overlay(server, {"port": overrides["port"]}, ServerConfig)
    if overrides.get("max_jobs") is not None:
        build = _overlay(build, {"max_jobs": overrides["max_jobs"]},
                         BuildConfig)

    # Derive project name from config path: "parent_dir/stem"
    if not project.name:
        project = _overlay(
            project,
            {"name": f"{path.parent.name}/{path.stem}"},
            ProjectConfig,
        )

    config = Config(
        project=project,
        build=build,
        server=server,
        bench=bench,
        runs=runs,
    )

    # Validate
    if not config.project.model:
        print("ERROR: 'project.model' is required in config", file=sys.stderr)
        sys.exit(1)
    if not config.runs:
        print("ERROR: 'runs' is required in config", file=sys.stderr)
        sys.exit(1)

    return config
