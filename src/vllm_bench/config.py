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
    isolate_flashinfer_cache: bool = False  # per-venv flashinfer JIT cache
    build_parallelism: int = 1   # concurrent branch builds (1 = sequential)
    compile_parallelism: int = 1  # concurrent server startups for JIT/CUDA graphs


@dataclass
class BuildConfig:
    use_precompiled: bool = True
    install_flash_attn: bool = False
    install_deepgemm: bool = False
    install_flashinfer_jit_cache: bool = False
    cuda_arch: str | None = None
    max_jobs: float = 1.0  # <=1: fraction of CPU cores, >1: absolute count
    torch_index: str = "https://download.pytorch.org/whl/cu130"
    auto_git_pull: bool = True  # git fetch+checkout before build


@dataclass
class ServerConfig:
    tp: int = 1
    max_model_len: int = 4096
    enforce_eager: bool = False
    gpu_memory_utilization: float | None = None
    attention_backend: str | None = None
    port: int = 8000
    compilation_config: dict | None = None
    kernel_config: dict | None = None   # --kernel-config (e.g. {"moe_backend": "triton"})
    log_level: str | None = None  # --log-level (e.g. "debug")
    clear_caches: bool = False    # wipe JIT caches before run
    env: dict[str, str] | None = None  # extra env vars for server process
    wait_timeout: int = 600  # seconds to wait for server health

    def build_serve_cmd(self, model: str, vllm_bin: str = "vllm") -> list[str]:
        """Build the vllm serve command list."""
        import json
        cmd = []
        if self.env:
            cmd += [f"{k}={v}" for k, v in self.env.items()]
        cmd += [
            vllm_bin, "serve", model,
            "--tensor-parallel-size", str(self.tp),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--port", str(self.port),
        ]
        if self.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        if self.attention_backend:
            cmd += ["--attention-backend", self.attention_backend]
        if self.compilation_config:
            cmd += ["-cc", json.dumps(self.compilation_config)]
        if self.kernel_config:
            cmd += ["--kernel-config", json.dumps(self.kernel_config)]
        return cmd


@dataclass
class BenchConfig:
    num_prompts: int = 1000
    input_len: int = 128
    output_len: int = 128
    request_rate: str = "inf"
    max_concurrency: int | None = None  # --max-concurrency (None = unlimited)
    warmup_prompts: int = 3   # server-level warmup (before bench tool)
    num_warmups: int = 50     # bench tool --num-warmups (JIT warmup)
    iterations: int = 1       # repeat benchmark N times per run


@dataclass
class EvalConfig:
    script: str = ""        # path relative to repo root
    args: str = ""          # additional CLI args (shlex.split'd)


@dataclass
class TestConfig:
    script: str = ""        # pytest target relative to repo root
    args: str = ""          # additional pytest CLI args (shlex.split'd)


@dataclass
class RunConfig:
    label: str
    branch: str  # set automatically from parent branch key
    commit: str = ""
    # Per-run overrides (server/bench only; build is set at branch level)
    server: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)


@dataclass
class BranchConfig:
    """Per-branch config: build overrides + runs."""
    build: dict = field(default_factory=dict)
    server: dict = field(default_factory=dict)
    bench: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    commit: str = ""
    runs: list[RunConfig] = field(default_factory=list)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    merged = dict(base)
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _overlay(base, overrides: dict, cls):
    """Create a new dataclass instance by overlaying overrides on base."""
    if not overrides:
        return base
    valid_keys = {f.name for f in dc_fields(cls)}
    unknown = set(overrides) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown override key(s) for {cls.__name__}: "
            f"{', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(valid_keys))}")
    kwargs = {f.name: getattr(base, f.name) for f in dc_fields(base)}
    for k, v in overrides.items():
        old = kwargs.get(k)
        if isinstance(old, dict) and isinstance(v, dict):
            kwargs[k] = _deep_merge(old, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    test: TestConfig = field(default_factory=TestConfig)
    branches: dict[str, BranchConfig] = field(default_factory=dict)
    runs: list[RunConfig] = field(default_factory=list)
    config_path: Path | None = field(default=None, repr=False)

    def _branch_config(self, run: RunConfig) -> BranchConfig:
        return self.branches.get(run.branch, BranchConfig())

    def effective_build(self, run: RunConfig) -> BuildConfig:
        """Global build + branch-level overrides. (No run-level build.)"""
        return _overlay(self.build, self._branch_config(run).build, BuildConfig)

    def effective_server(self, run: RunConfig) -> ServerConfig:
        """Global -> branch -> run server config."""
        base = _overlay(self.server, self._branch_config(run).server,
                        ServerConfig)
        return _overlay(base, run.server, ServerConfig)

    def effective_bench(self, run: RunConfig) -> BenchConfig:
        """Global -> branch -> run bench config."""
        base = _overlay(self.bench, self._branch_config(run).bench,
                        BenchConfig)
        return _overlay(base, run.bench, BenchConfig)

    def effective_eval(self, run: RunConfig) -> EvalConfig:
        """Global -> branch -> run eval config."""
        base = _overlay(self.eval, self._branch_config(run).eval,
                        EvalConfig)
        return _overlay(base, run.eval, EvalConfig)

    def effective_test(self, run: RunConfig) -> TestConfig:
        """Global -> branch -> run test config."""
        base = _overlay(self.test, self._branch_config(run).test,
                        TestConfig)
        return _overlay(base, run.test, TestConfig)


# ── YAML loading ─────────────────────────────────────────────────────

_SECTIONS = {"project", "build", "server", "bench", "eval", "test", "branches"}

_SECTION_FIELDS = {
    "project": {f.name for f in dc_fields(ProjectConfig)},
    "build": {f.name for f in dc_fields(BuildConfig)},
    "server": {f.name for f in dc_fields(ServerConfig)},
    "bench": {f.name for f in dc_fields(BenchConfig)},
    "eval": {f.name for f in dc_fields(EvalConfig)},
    "test": {f.name for f in dc_fields(TestConfig)},
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


_BRANCH_FIELDS = {"build", "server", "bench", "eval", "test", "commit", "runs"}
_RUN_FIELDS = {"label", "commit", "server", "bench", "eval", "test"}


def _parse_branches(raw_branches: dict) -> tuple[dict[str, BranchConfig],
                                                   list[RunConfig]]:
    """Parse branches section, returning branch configs and flattened run list."""
    branch_configs: dict[str, BranchConfig] = {}
    all_runs: list[RunConfig] = []

    for branch_name, raw in raw_branches.items():
        if not isinstance(raw, dict):
            raw = {}
        unknown = set(raw) - _BRANCH_FIELDS
        if unknown:
            warnings.warn(
                f"branches[{branch_name!r}]: unknown key(s): "
                f"{', '.join(sorted(unknown))!r}")

        branch_commit = raw.get("commit", "")
        raw_runs = raw.get("runs") or []
        runs: list[RunConfig] = []
        for i, r in enumerate(raw_runs):
            if isinstance(r, str):
                # Short form: just a label string
                r = {"label": r}
            run_unknown = set(r) - _RUN_FIELDS
            if run_unknown:
                warnings.warn(
                    f"branches[{branch_name!r}].runs[{i}] "
                    f"({r.get('label', '?')}): unknown key(s): "
                    f"{', '.join(sorted(run_unknown))!r}")
            try:
                runs.append(RunConfig(
                    label=r["label"],
                    branch=branch_name,
                    commit=r.get("commit", branch_commit),
                    server=r.get("server") or {},
                    bench=r.get("bench") or {},
                    eval=r.get("eval") or {},
                    test=r.get("test") or {},
                ))
            except KeyError as e:
                raise ValueError(
                    f"branches[{branch_name!r}].runs[{i}]: "
                    f"missing required field {e}") from e

        branch_configs[branch_name] = BranchConfig(
            build=raw.get("build") or {},
            server=raw.get("server") or {},
            bench=raw.get("bench") or {},
            eval=raw.get("eval") or {},
            test=raw.get("test") or {},
            commit=branch_commit,
            runs=runs,
        )
        all_runs.extend(runs)

    return branch_configs, all_runs


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
    eval_ = _build_section(EvalConfig, raw.get("eval") or {}, "eval")
    test = _build_section(TestConfig, raw.get("test") or {}, "test")
    branches, runs = _parse_branches(raw.get("branches") or {})

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
        eval=eval_,
        test=test,
        branches=branches,
        runs=runs,
        config_path=path,
    )

    # Validate
    if not config.project.model:
        print("ERROR: 'project.model' is required in config", file=sys.stderr)
        sys.exit(1)
    if not config.runs:
        print("ERROR: 'runs' is required in config", file=sys.stderr)
        sys.exit(1)

    return config
