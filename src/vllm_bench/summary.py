# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse benchmark results and generate comparison tables."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .resolved import ResolvedRun

# Metrics grouped by section, with divider labels matching vllm bench output.
# Each entry is (section_label | None, metric_name, regex_pattern).
METRIC_SECTIONS: list[tuple[str | None, str, str]] = [
    (None, "Successful requests",        r"Successful requests:\s*([\d.]+)"),
    (None, "Failed requests",            r"Failed requests:\s*([\d.]+)"),
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
    lines.append("  RESULTS SUMMARY")
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

    # Per-run details with effective config
    lines.append("Runs:")
    for r in resolved_runs:
        lines.append(f"  - {r.label}:")
        lines.append(f"      branch:     {r.branch}"
                     f"{f' @ {r.commit}' if r.commit else ''}")
        lines.append(f"      TP:         {r.server.tp}")
        lines.append(f"      max_len:    {r.server.max_model_len}")
        lines.append(f"      prompts:    {r.bench.num_prompts} "
                     f"(in={r.bench.input_len}, out={r.bench.output_len})")
        lines.append(f"      precompiled: {r.build.use_precompiled}")
        if r.server.compilation_config:
            lines.append(f"      cc:         "
                         f"{json.dumps(r.server.compilation_config)}")
        if r.server.enforce_eager:
            lines.append(f"      eager:      True")
        if r.server.gpu_memory_utilization is not None:
            lines.append(f"      gpu_mem:    {r.server.gpu_memory_utilization}")
        if r.build.cuda_arch:
            lines.append(f"      cuda_arch:  {r.build.cuda_arch}")
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
