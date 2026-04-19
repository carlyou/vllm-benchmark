# SPDX-License-Identifier: Apache-2.0
"""Tests for config loading, merging, and hierarchy resolution."""

import warnings
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from vllm_bench.config import (
    BenchConfig,
    BranchConfig,
    BuildConfig,
    Config,
    EvalConfig,
    ProjectConfig,
    RunConfig,
    ServerConfig,
    TestConfig,
    _deep_merge,
    _overlay,
    _parse_branches,
    load_config,
)


# ── _deep_merge ─────────────────────────────────────────────────────


class TestDeepMerge:
    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_value(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}}
        over = {"a": {"y": 3, "z": 4}}
        assert _deep_merge(base, over) == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_nested_override_with_non_dict(self):
        assert _deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}

    def test_empty_override(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == base

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"y": 2}})
        assert base == {"a": {"x": 1}}


# ── _overlay ────────────────────────────────────────────────────────


class TestOverlay:
    def test_no_overrides_returns_base(self):
        base = ServerConfig(tp=4)
        assert _overlay(base, {}, ServerConfig) is base

    def test_override_scalar(self):
        base = ServerConfig(tp=1)
        result = _overlay(base, {"tp": 4}, ServerConfig)
        assert result.tp == 4

    def test_override_dict_merges(self):
        base = ServerConfig(compilation_config={"a": 1, "b": 2})
        result = _overlay(base, {"compilation_config": {"b": 3, "c": 4}},
                          ServerConfig)
        assert result.compilation_config == {"a": 1, "b": 3, "c": 4}

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown override key"):
            _overlay(ServerConfig(), {"bogus": 42}, ServerConfig)

    def test_does_not_mutate_base(self):
        base = ServerConfig(tp=1)
        _overlay(base, {"tp": 4}, ServerConfig)
        assert base.tp == 1


# ── ServerConfig.build_serve_cmd / format_serve_cmd ─────────────────


class TestServerConfigCmd:
    def test_basic_cmd(self):
        srv = ServerConfig(tp=4, max_model_len=2048, port=9000)
        cmd = srv.build_serve_cmd("my-model")
        assert cmd[:3] == ["vllm", "serve", "my-model"]
        assert "--tensor-parallel-size" in cmd
        assert "4" in cmd
        assert "--max-model-len" in cmd
        assert "2048" in cmd
        assert "--port" in cmd
        assert "9000" in cmd

    def test_custom_vllm_bin(self):
        srv = ServerConfig()
        cmd = srv.build_serve_cmd("m", vllm_bin="/opt/vllm")
        assert cmd[0] == "/opt/vllm"

    def test_gpu_memory_utilization(self):
        srv = ServerConfig(gpu_memory_utilization=0.9)
        cmd = srv.build_serve_cmd("m")
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    def test_no_gpu_memory_utilization(self):
        srv = ServerConfig()
        cmd = srv.build_serve_cmd("m")
        assert "--gpu-memory-utilization" not in cmd

    def test_enforce_eager(self):
        srv = ServerConfig(enforce_eager=True)
        assert "--enforce-eager" in srv.build_serve_cmd("m")

    def test_compilation_config(self):
        srv = ServerConfig(compilation_config={"mode": "NONE"})
        cmd = srv.build_serve_cmd("m")
        assert "-cc" in cmd

    def test_kernel_config(self):
        srv = ServerConfig(kernel_config={"moe_backend": "triton"})
        cmd = srv.build_serve_cmd("m")
        assert "--kernel-config" in cmd

    def test_env_not_in_build_cmd(self):
        srv = ServerConfig(env={"FOO": "1"})
        cmd = srv.build_serve_cmd("m")
        assert "FOO=1" not in cmd

    def test_env_in_format_cmd(self):
        srv = ServerConfig(env={"VLLM_USE_DEEP_GEMM": "0"})
        s = srv.format_serve_cmd("m")
        assert s.startswith("VLLM_USE_DEEP_GEMM=0 vllm serve")

    def test_format_cmd_no_env(self):
        srv = ServerConfig()
        s = srv.format_serve_cmd("m")
        assert s.startswith("vllm serve m")


# ── _parse_branches ─────────────────────────────────────────────────


class TestParseBranches:
    def test_single_branch_single_run(self):
        raw = {
            "main": {
                "runs": [{"label": "baseline"}]
            }
        }
        branches, runs = _parse_branches(raw)
        assert "main" in branches
        assert len(runs) == 1
        assert runs[0].label == "baseline"
        assert runs[0].branch == "main"

    def test_short_form_run(self):
        raw = {"main": {"runs": ["baseline"]}}
        _, runs = _parse_branches(raw)
        assert runs[0].label == "baseline"

    def test_branch_commit_inherited(self):
        raw = {
            "feat": {
                "commit": "abc123",
                "runs": [{"label": "test"}]
            }
        }
        _, runs = _parse_branches(raw)
        assert runs[0].commit == "abc123"

    def test_run_commit_overrides_branch(self):
        raw = {
            "feat": {
                "commit": "abc123",
                "runs": [{"label": "test", "commit": "def456"}]
            }
        }
        _, runs = _parse_branches(raw)
        assert runs[0].commit == "def456"

    def test_missing_label_raises(self):
        raw = {"main": {"runs": [{"server": {"tp": 4}}]}}
        with pytest.raises(ValueError, match="missing required field"):
            _parse_branches(raw)

    def test_per_run_server_override(self):
        raw = {
            "main": {
                "runs": [{
                    "label": "fast",
                    "server": {"tp": 8}
                }]
            }
        }
        _, runs = _parse_branches(raw)
        assert runs[0].server == {"tp": 8}

    def test_empty_branch_value(self):
        raw = {"main": None}
        branches, runs = _parse_branches(raw)
        assert "main" in branches
        assert len(runs) == 0

    def test_unknown_branch_key_warns(self):
        raw = {"main": {"bogus": True, "runs": [{"label": "x"}]}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _parse_branches(raw)
            assert any("unknown key" in str(x.message) for x in w)

    def test_multiple_branches(self):
        raw = {
            "main": {"runs": [{"label": "a"}]},
            "feat": {"runs": [{"label": "b"}, {"label": "c"}]},
        }
        branches, runs = _parse_branches(raw)
        assert len(branches) == 2
        assert len(runs) == 3


# ── Config hierarchy resolution ─────────────────────────────────────


class TestConfigHierarchy:
    def _make_config(self):
        return Config(
            server=ServerConfig(tp=1, max_model_len=4096),
            branches={
                "feat": BranchConfig(
                    server={"tp": 4},
                    runs=[
                        RunConfig(label="base", branch="feat"),
                        RunConfig(label="fast", branch="feat",
                                  server={"max_model_len": 2048}),
                    ],
                ),
            },
            runs=[
                RunConfig(label="base", branch="feat"),
                RunConfig(label="fast", branch="feat",
                          server={"max_model_len": 2048}),
            ],
        )

    def test_global_defaults(self):
        config = self._make_config()
        run = config.runs[0]
        srv = config.effective_server(run)
        # Branch overrides tp
        assert srv.tp == 4
        # Global default preserved
        assert srv.max_model_len == 4096

    def test_run_overrides_branch(self):
        config = self._make_config()
        run = config.runs[1]  # "fast" run
        srv = config.effective_server(run)
        # Branch level
        assert srv.tp == 4
        # Run level overrides global
        assert srv.max_model_len == 2048

    def test_unknown_branch_uses_defaults(self):
        config = Config(server=ServerConfig(tp=2))
        run = RunConfig(label="x", branch="nonexistent")
        srv = config.effective_server(run)
        assert srv.tp == 2

    def test_build_no_run_level(self):
        config = Config(
            build=BuildConfig(cuda_arch="9.0"),
            branches={
                "feat": BranchConfig(build={"use_precompiled": False})
            },
        )
        run = RunConfig(label="x", branch="feat")
        b = config.effective_build(run)
        assert b.cuda_arch == "9.0"
        assert b.use_precompiled is False


# ── load_config (end-to-end) ────────────────────────────────────────


class TestLoadConfig:
    def test_minimal_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: my-model
            branches:
              main:
                runs:
                  - label: baseline
        """))
        config = load_config(str(cfg))
        assert config.project.model == "my-model"
        assert len(config.runs) == 1
        assert config.runs[0].label == "baseline"

    def test_project_name_derived(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: m
            branches:
              main:
                runs:
                  - label: x
        """))
        config = load_config(str(cfg))
        assert config.project.name == f"{tmp_path.name}/test"

    def test_port_override(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: m
            branches:
              main:
                runs:
                  - label: x
        """))
        config = load_config(str(cfg), port=9999)
        assert config.server.port == 9999

    def test_missing_model_exits(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            branches:
              main:
                runs:
                  - label: x
        """))
        with pytest.raises(SystemExit):
            load_config(str(cfg))

    def test_missing_runs_exits(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: m
        """))
        with pytest.raises(SystemExit):
            load_config(str(cfg))

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit):
            load_config("/nonexistent/path.yaml")

    def test_unknown_top_level_warns(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: m
            branches:
              main:
                runs:
                  - label: x
            bogus_section: true
        """))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(str(cfg))
            assert any("Unknown top-level" in str(x.message) for x in w)

    def test_work_dir_expanded(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              model: m
              work_dir: ~/my-bench
            branches:
              main:
                runs:
                  - label: x
        """))
        config = load_config(str(cfg))
        assert "~" not in config.project.work_dir

    def test_full_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(dedent("""\
            project:
              repo: https://github.com/test/vllm.git
              model: test-model
            build:
              cuda_arch: "10.0"
              torch_index: https://download.pytorch.org/whl/cu130
              install_flashinfer_jit_cache: true
            server:
              tp: 4
              max_model_len: 4096
              gpu_memory_utilization: 0.95
              env:
                VLLM_USE_DEEP_GEMM: "0"
              kernel_config:
                moe_backend: triton
              compilation_config:
                cudagraph_mode: NONE
                custom_ops: ["+quant_fp8"]
            branches:
              main:
                runs:
                  - label: baseline
                  - label: fuse_on
                    server:
                      compilation_config:
                        pass_config:
                          fuse_attn_quant: true
        """))
        config = load_config(str(cfg))
        assert config.project.repo == "https://github.com/test/vllm.git"
        assert config.build.install_flashinfer_jit_cache is True
        assert config.server.env == {"VLLM_USE_DEEP_GEMM": "0"}
        assert config.server.kernel_config == {"moe_backend": "triton"}
        assert len(config.runs) == 2

        # Check hierarchy: fuse_on run merges compilation_config
        fuse_run = config.runs[1]
        srv = config.effective_server(fuse_run)
        assert srv.compilation_config["custom_ops"] == ["+quant_fp8"]
        assert srv.compilation_config["pass_config"] == {
            "fuse_attn_quant": True
        }
