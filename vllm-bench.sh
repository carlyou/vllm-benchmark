#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Thin wrapper around the Python CLI.
# Usage: ./vllm-bench.sh <command> configs/mla_quant_fusion/h100_fp8.yaml [options]

set -euo pipefail
exec uv run vllm-bench "$@"
