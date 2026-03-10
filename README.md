# vLLM A/B Benchmark

Reusable harness for comparing vLLM branches side-by-side. Define runs in YAML — the tool handles cloning, building, server lifecycle, and result collection.

## Usage

```bash
# Full pipeline: build + benchmark
./vllm-bench.sh run configs/mla_quant_fusion/h100_fp8.yaml

# Individual steps
./vllm-bench.sh build configs/mla_quant_fusion/h100_fp8.yaml
./vllm-bench.sh bench configs/mla_quant_fusion/h100_fp8.yaml

# Filter to specific runs
./vllm-bench.sh run configs/mla_quant_fusion/h100_fp8.yaml --run baseline --run candidate

# Override build/server settings
./vllm-bench.sh run configs/my_test.yaml --port 9000 --max-jobs 8
```

Force rebuild via environment variable:

```bash
FORCE_BUILD=1 ./vllm-bench.sh run configs/my_test.yaml
```

## Config

```yaml
project:
  repo: https://github.com/vllm-project/vllm.git
  model: meta-llama/Llama-3.1-8B-Instruct

build:
  use_precompiled: true
  cuda_arch: "9.0"      # optional, auto-detected if omitted

server:
  tp: 1
  max_model_len: 4096

bench:
  num_prompts: 200
  input_len: 128
  output_len: 128

runs:
  - label: baseline
    branch: main
  - label: candidate
    branch: my-optimization
    server:                    # per-run overrides
      compilation_config:
        pass_config:
          fuse_attn_quant: true
```

Each run clones the specified branch, builds vLLM (with caching), starts a server, benchmarks with `vllm bench serve`, and collects throughput/TTFT/TPOT metrics.

## Subcommands

| Command | Description |
|---|---|
| `run` | Build + benchmark (full pipeline) |
| `build` | Clone and build all branches |
| `bench` | Benchmark only (builds must already exist) |
| `compile` | Pre-compile CUDA graphs (start/check/stop) |

## Environment Variables

| Variable | Description |
|---|---|
| `FORCE_BUILD` | `1` to rebuild even if cached |

## Requirements

- Python 3.12+, [uv](https://docs.astral.sh/uv/), CUDA toolkit
