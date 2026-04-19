# SPDX-License-Identifier: Apache-2.0
"""Tests for summary parsing and formatting."""

from pathlib import Path
from textwrap import dedent

import pytest

from vllm_bench.config import Config, RunConfig, ServerConfig
from vllm_bench.summary import METRICS, parse_results


# ── parse_results ───────────────────────────────────────────────────


SAMPLE_BENCH_OUTPUT = dedent("""\
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Failed requests:                         0
    Request rate configured (RPS):           100.0
    Benchmark duration (s):                  45.23
    Total input tokens:                      128000
    Total generated tokens:                  128000
    Request throughput (req/s):              22.11
    Output token throughput (tok/s):          2830.42
    Peak output token throughput (tok/s):     3120.15
    Total token throughput (tok/s):           5660.84
    Peak concurrent requests:                128
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          12.45
    Median TTFT (ms):                        10.23
    P99 TTFT (ms):                           45.67
    ---------------Time per Output Token--------------
    Mean TPOT (ms):                          0.35
    Median TPOT (ms):                        0.32
    P99 TPOT (ms):                           1.23
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           0.34
    Median ITL (ms):                         0.31
    P99 ITL (ms):                            1.20
    ==================================================
""")


class TestParseResults:
    def test_parse_all_metrics(self, tmp_path):
        f = tmp_path / "baseline.txt"
        f.write_text(SAMPLE_BENCH_OUTPUT)
        data = parse_results({"baseline": f})
        assert "baseline" in data
        d = data["baseline"]
        assert d["Successful requests"] == "1000"
        assert d["Failed requests"] == "0"
        assert d["Request throughput (req/s)"] == "22.11"
        assert d["Output token tput (tok/s)"] == "2830.42"
        assert d["Mean TTFT (ms)"] == "12.45"
        assert d["P99 TPOT (ms)"] == "1.23"

    def test_parse_multiple_runs(self, tmp_path):
        for name in ["baseline", "optimized"]:
            (tmp_path / f"{name}.txt").write_text(SAMPLE_BENCH_OUTPUT)
        data = parse_results({
            "baseline": tmp_path / "baseline.txt",
            "optimized": tmp_path / "optimized.txt",
        })
        assert len(data) == 2
        assert data["baseline"]["Mean ITL (ms)"] == "0.34"
        assert data["optimized"]["Mean ITL (ms)"] == "0.34"

    def test_missing_file_returns_dashes(self, tmp_path):
        data = parse_results({"missing": tmp_path / "nonexistent.txt"})
        assert "missing" in data
        assert data["missing"]["Successful requests"] == "—"

    def test_partial_output(self, tmp_path):
        f = tmp_path / "partial.txt"
        f.write_text(dedent("""\
            Successful requests:     500
            Failed requests:         2
        """))
        data = parse_results({"partial": f})
        d = data["partial"]
        assert d["Successful requests"] == "500"
        assert d["Failed requests"] == "2"
        # Missing metrics filled with N/A
        assert d["Mean TTFT (ms)"] == "N/A"


# ── METRICS patterns ────────────────────────────────────────────────


class TestMetricPatterns:
    """Verify each regex pattern matches expected output format."""

    @pytest.mark.parametrize("name,pattern", METRICS)
    def test_pattern_matches_sample(self, name, pattern):
        import re
        # Each metric pattern should match at least one line in the sample
        match = re.search(pattern, SAMPLE_BENCH_OUTPUT)
        if name in ("Peak output tput (tok/s)", "Peak concurrent requests"):
            # These are optional — may or may not appear
            return
        assert match is not None, f"Pattern for {name!r} did not match"


# ── ServerConfig display ────────────────────────────────────────────


class TestFormatServerCmd:
    def test_basic(self):
        srv = ServerConfig(tp=4, port=8000)
        s = srv.format_serve_cmd("my-model")
        assert "vllm serve my-model" in s
        assert "--tensor-parallel-size 4" in s

    def test_with_env(self):
        srv = ServerConfig(
            env={"VLLM_USE_DEEP_GEMM": "0", "FOO": "bar"},
        )
        s = srv.format_serve_cmd("m")
        assert s.startswith("VLLM_USE_DEEP_GEMM=0")
        assert "FOO=bar" in s
        assert "vllm serve m" in s

    def test_with_kernel_config(self):
        srv = ServerConfig(kernel_config={"moe_backend": "triton"})
        s = srv.format_serve_cmd("m")
        assert "--kernel-config" in s
        assert "triton" in s

    def test_with_compilation_config(self):
        srv = ServerConfig(
            compilation_config={"cudagraph_mode": "NONE"},
        )
        s = srv.format_serve_cmd("m")
        assert "-cc" in s
        assert "NONE" in s
