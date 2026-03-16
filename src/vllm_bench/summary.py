# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse benchmark/eval results and generate comparison tables."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Config
from .resolved import ResolvedRun

# Metrics grouped by section, with divider labels matching vllm bench output.
# Each entry is (section_label | None, metric_name, regex_pattern).
METRIC_SECTIONS: list[tuple[str | None, str, str]] = [
    (None, "Successful requests",        r"Successful requests:\s*([\d.]+)"),
    (None, "Failed requests",            r"Failed requests:\s*([\d.]+)"),
    (None, "Request rate (RPS)",         r"Request rate configured \(RPS\):\s*([\d.]+)"),
    (None, "Benchmark duration (s)",     r"Benchmark duration \(s\):\s*([\d.]+)"),
    (None, "Total input tokens",         r"Total input tokens:\s*([\d.]+)"),
    (None, "Total generated tokens",     r"Total generated tokens:\s*([\d.]+)"),
    (None, "Request throughput (req/s)", r"Request throughput.*?:\s*([\d.]+)"),
    (None, "Output token tput (tok/s)",  r"Output token throughput.*?:\s*([\d.]+)"),
    (None, "Peak output tput (tok/s)",   r"Peak output token throughput.*?:\s*([\d.]+)"),
    (None, "Total token tput (tok/s)",   r"Total token throughput.*?:\s*([\d.]+)"),
    (None, "Peak concurrent requests",   r"Peak concurrent requests:\s*([\d.]+)"),
    ("Time to First Token",  "Mean TTFT (ms)",   r"Mean TTFT.*?:\s*([\d.]+)"),
    (None,                    "Median TTFT (ms)", r"Median TTFT.*?:\s*([\d.]+)"),
    (None,                    "P99 TTFT (ms)",    r"P99 TTFT.*?:\s*([\d.]+)"),
    ("Time per Output Token", "Mean TPOT (ms)",   r"Mean TPOT.*?:\s*([\d.]+)"),
    (None,                    "Median TPOT (ms)", r"Median TPOT.*?:\s*([\d.]+)"),
    (None,                    "P99 TPOT (ms)",    r"P99 TPOT.*?:\s*([\d.]+)"),
    ("Inter-token Latency",   "Mean ITL (ms)",    r"Mean ITL.*?:\s*([\d.]+)"),
    (None,                    "Median ITL (ms)",  r"Median ITL.*?:\s*([\d.]+)"),
    (None,                    "P99 ITL (ms)",     r"P99 ITL.*?:\s*([\d.]+)"),
]

# Flat list for parsing
METRICS = [(name, pattern) for _, name, pattern in METRIC_SECTIONS]


def parse_results(result_files: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Parse benchmark output files into structured data."""
    data: dict[str, dict[str, str]] = {}
    for label, fpath in result_files.items():
        vals: dict[str, str] = {}
        if fpath.exists():
            content = fpath.read_text()
            for name, pattern in METRICS:
                m = re.search(pattern, content)
                vals[name] = m.group(1) if m else "N/A"
        else:
            for name, _ in METRICS:
                vals[name] = "—"
        data[label] = vals
    return data


def _get_hardware_info(venv_python: Path) -> dict[str, str]:
    """Collect hardware/runtime info."""
    info: dict[str, str] = {}

    if shutil.which("nvidia-smi"):
        try:
            info["GPU"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name",
                 "--format=csv,noheader"],
                text=True,
            ).strip().split("\n")[0]
        except Exception:
            info["GPU"] = "N/A"
        try:
            info["GPU mem"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader"],
                text=True,
            ).strip().split("\n")[0]
        except Exception:
            info["GPU mem"] = "N/A"
    else:
        info["GPU"] = "N/A"
        info["GPU mem"] = "N/A"

    python = str(venv_python)
    for label, cmd in [
        ("CUDA", [python, "-c", "import torch; print(torch.version.cuda)"]),
        ("PyTorch", [python, "-c", "import torch; print(torch.__version__)"]),
        ("Python", [python, "--version"]),
    ]:
        try:
            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if label == "Python":
                out = out.replace("Python ", "")
            info[label] = out
        except Exception:
            info[label] = "N/A"

    info["Platform"] = platform.machine()
    return info


def _format_server_cmd(config: Config, r: ResolvedRun) -> str:
    """Reconstruct the vllm serve command for display."""
    srv = r.server
    parts = [
        "vllm serve", config.project.model,
        "--tensor-parallel-size", str(srv.tp),
        "--max-model-len", str(srv.max_model_len),
        "--trust-remote-code",
        "--port", str(srv.port),
    ]
    if srv.gpu_memory_utilization is not None:
        parts += ["--gpu-memory-utilization", str(srv.gpu_memory_utilization)]
    if srv.enforce_eager:
        parts += ["--enforce-eager"]
    if srv.compilation_config:
        parts += ["-cc", json.dumps(srv.compilation_config)]
    return " ".join(parts)


def _format_bench_cmd(config: Config, r: ResolvedRun) -> str:
    """Reconstruct the vllm bench serve command for display."""
    bench = r.bench
    parts = [
        "vllm bench serve",
        "--model", config.project.model,
        "--num-prompts", str(bench.num_prompts),
        "--request-rate", str(bench.request_rate),
        "--random-input-len", str(bench.input_len),
        "--random-output-len", str(bench.output_len),
        "--num-warmups", str(bench.num_warmups),
        "--ignore-eos",
    ]
    if bench.max_concurrency is not None:
        parts += ["--max-concurrency", str(bench.max_concurrency)]
    return " ".join(parts)


def _format_eval_cmd(config: Config, r: ResolvedRun) -> str:
    """Reconstruct the eval script command for display."""
    eval_cfg = r.eval
    if not eval_cfg.script:
        return ""
    parts = ["python", eval_cfg.script]
    if eval_cfg.args:
        parts.append(eval_cfg.args)
    return " ".join(parts)


def format_summary(config: Config,
                   result_files: dict[str, Path],
                   resolved_runs: list[ResolvedRun],
                   repos_dir: Path) -> str:
    """Generate the full summary text."""
    data = parse_results(result_files)
    labels = list(result_files.keys())
    proj = config.project

    lines: list[str] = []
    lines.append("=" * 44)
    lines.append("  BENCHMARK SUMMARY")
    lines.append("=" * 44)
    lines.append("")
    lines.append(f"Config:      {proj.name}")
    lines.append(f"Model:       {proj.model}")
    lines.append(f"Repo:        {proj.repo}")
    lines.append(f"Work dir:    {proj.work_dir}")
    lines.append("")

    # Hardware info from first run's venv
    first = resolved_runs[0]
    hw = _get_hardware_info(first.venv_python)
    lines.append("Hardware/Runtime:")
    for k, v in hw.items():
        lines.append(f"  {k + ':':11s}{v}")
    lines.append("")

    # Per-run details with commands
    lines.append("Runs:")
    for r in resolved_runs:
        lines.append(f"  - {r.label}:")
        lines.append(f"      branch:     {r.branch}"
                     f"{f' @ {r.commit}' if r.commit else ''}")
        lines.append(f"      server:     $ {_format_server_cmd(config, r)}")
        lines.append(f"      bench:      $ {_format_bench_cmd(config, r)}")
    lines.append("")

    # Markdown metrics table
    metric_w = max(len(name) for _, name, _ in METRIC_SECTIONS)
    val_ws = [max(len(label), 10) for label in labels]

    # Header
    header = f"| {'Metric':<{metric_w}} |"
    sep = f"| {'-' * metric_w} |"
    for i, label in enumerate(labels):
        header += f" {label:>{val_ws[i]}} |"
        sep += f" {'-' * val_ws[i]}: |"
    lines.append(header)
    lines.append(sep)

    # Rows with section dividers
    for section, name, _ in METRIC_SECTIONS:
        if section is not None:
            divider = f"| **{section}** |"
            for w in val_ws:
                divider += f" {'':{w}} |"
            lines.append(divider)
        row = f"| {name:<{metric_w}} |"
        for i, label in enumerate(labels):
            row += f" {data[label][name]:>{val_ws[i]}} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _load_eval_result(path: Path) -> dict[str, Any]:
    """Load eval result JSON, returning empty dict on failure."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


_EVAL_METRICS: list[tuple[str, str, str]] = [
    # (column header, json key, format)
    ("Accuracy", "accuracy", ".3f"),
    ("Invalid", "invalid_rate", ".3f"),
    ("Questions", "num_questions", "d"),
    ("Latency (s)", "latency", ".2f"),
    ("Q/s", "questions_per_second", ".2f"),
    ("Tokens", "total_output_tokens", "d"),
    ("Tok/s", "tokens_per_second", ".1f"),
]


def format_eval_summary(config: Config,
                        result_files: dict[str, Path],
                        resolved_runs: list[ResolvedRun]) -> str:
    """Generate eval summary table from saved result JSONs."""
    proj = config.project

    lines: list[str] = []
    lines.append("=" * 44)
    lines.append("  EVAL SUMMARY")
    lines.append("=" * 44)
    lines.append("")
    lines.append(f"Config:      {proj.name}")
    lines.append(f"Model:       {proj.model}")
    lines.append(f"Repo:        {proj.repo}")
    lines.append("")

    # Hardware info from first run's venv
    first = resolved_runs[0]
    hw = _get_hardware_info(first.venv_python)
    lines.append("Hardware/Runtime:")
    for k, v in hw.items():
        lines.append(f"  {k + ':':11s}{v}")
    lines.append("")

    # Per-run details with commands
    lines.append("Runs:")
    for r in resolved_runs:
        lines.append(f"  - {r.label}:")
        lines.append(f"      branch:     {r.branch}"
                     f"{f' @ {r.commit}' if r.commit else ''}")
        lines.append(f"      server:     $ {_format_server_cmd(config, r)}")
        eval_cmd = _format_eval_cmd(config, r)
        if eval_cmd:
            lines.append(f"      eval:       $ {eval_cmd}")
    lines.append("")

    # Load results
    results: list[tuple[str, dict[str, Any]]] = []
    for r in resolved_runs:
        if r.label not in result_files:
            continue
        data = _load_eval_result(result_files[r.label])
        results.append((r.label, data))

    if not results:
        lines.append("No eval results found.")
        lines.append("")
        return "\n".join(lines)

    # Build formatted rows
    rows: list[dict[str, str]] = []
    for label, data in results:
        row = {"Run": label}
        for col, key, fmt in _EVAL_METRICS:
            val = data.get(key)
            if val is not None:
                row[col] = f"{val:{fmt}}"
            else:
                row[col] = "—"
        rows.append(row)

    # Calculate column widths
    cols = ["Run"] + [col for col, _, _ in _EVAL_METRICS]
    widths = {c: max(len(c), *(len(row[c]) for row in rows)) for c in cols}

    # Header
    header = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    lines.append(header)
    lines.append(sep)

    # Data rows
    for row in rows:
        align = {c: f"{row[c]:>{widths[c]}}" for c in cols}
        align["Run"] = f"{row['Run']:<{widths['Run']}}"
        line = "| " + " | ".join(align[c] for c in cols) + " |"
        lines.append(line)

    lines.append("")
    return "\n".join(lines)
