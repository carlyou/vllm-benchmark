# SPDX-License-Identifier: Apache-2.0
"""Tests for resolved run configuration."""

from pathlib import Path

from vllm_bench.config import (
    BenchConfig,
    BuildConfig,
    EvalConfig,
    RunConfig,
    ServerConfig,
    TestConfig,
)
from vllm_bench.resolved import ResolvedRun


class TestResolvedRun:
    def _make_run(self, **kwargs):
        run = kwargs.pop("run", RunConfig(
            label=kwargs.pop("label", "test"),
            branch=kwargs.pop("branch", "main"),
            commit=kwargs.pop("commit", "abc123"),
        ))
        defaults = dict(
            run=run,
            server=ServerConfig(tp=4),
            build=BuildConfig(),
            bench=BenchConfig(),
            eval=EvalConfig(),
            test=TestConfig(),
            repo_dir=Path("/tmp/repo"),
            venv_python=Path("/tmp/repo/.venv/bin/python"),
            vllm_bin=Path("/tmp/repo/.venv/bin/vllm"),
        )
        defaults.update(kwargs)
        return ResolvedRun(**defaults)

    def test_properties(self):
        r = self._make_run()
        assert r.label == "test"
        assert r.branch == "main"
        assert r.commit == "abc123"

    def test_with_server(self):
        r = self._make_run()
        r2 = r.with_server(tp=8)
        assert r2.server.tp == 8
        # Original unchanged
        assert r.server.tp == 4

    def test_with_server_preserves_other_fields(self):
        r = self._make_run(label="original")
        r2 = r.with_server(max_model_len=2048)
        assert r2.label == "original"
        assert r2.server.max_model_len == 2048
        assert r2.server.tp == 4  # Preserved from original
