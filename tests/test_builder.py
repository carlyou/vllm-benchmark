# SPDX-License-Identifier: Apache-2.0
"""Tests for builder utilities."""

import multiprocessing

import pytest

from vllm_bench.builder import _resolve_jobs, branch_to_dir


class TestBranchToDir:
    def test_simple_branch(self):
        assert branch_to_dir("main", "") == "main"

    def test_slash_replaced(self):
        assert branch_to_dir("feat/my-feature", "") == "feat--my-feature"

    def test_with_commit(self):
        result = branch_to_dir("main", "abc123")
        assert "main" in result
        assert "abc123" in result

    def test_double_dash_separator(self):
        result = branch_to_dir("user/repo/branch", "")
        assert "/" not in result


class TestResolveJobs:
    def test_absolute_count(self):
        assert _resolve_jobs(4) == 4
        assert _resolve_jobs(8.0) == 8

    def test_fraction_of_cores(self):
        ncpu = multiprocessing.cpu_count()
        result = _resolve_jobs(0.5)
        assert result == max(1, int(ncpu * 0.5))

    def test_fraction_min_one(self):
        result = _resolve_jobs(0.001)
        assert result >= 1

    def test_one_means_all_cores(self):
        ncpu = multiprocessing.cpu_count()
        assert _resolve_jobs(1.0) == ncpu
