# vLLM A/B Benchmark

Reusable harness for comparing vLLM branches side-by-side. Define runs in YAML — the script handles cloning, building, server lifecycle, and result collection.

## Usage

```bash
./benchmark.sh configs/branch_compare.yaml
```

Override settings via environment variables:

```bash
CUDA_ARCH=12.1 FORCE_BUILD=1 ./benchmark.sh configs/my_test.yaml
```

## Config

```yaml
repo: https://github.com/vllm-project/vllm.git
model: meta-llama/Llama-3.1-8B-Instruct
tp: 1
max_model_len: 4096
num_prompts: 200
input_len: 128
output_len: 128
use_precompiled: true

runs:
  - label: baseline
    branch: main
  - label: candidate
    branch: my-optimization
    compilation_config:
      pass_config:
        fuse_attn_quant: true
```

Each run clones the specified branch, builds vLLM (with caching), starts a server, benchmarks with `vllm bench serve`, and collects throughput/TTFT/TPOT metrics.

## Environment Variables

| Variable | Description |
|---|---|
| `CUDA_ARCH` | Target architecture (e.g. `12.1`) |
| `FORCE_BUILD` | `1` to rebuild even if cached |
| `DEBUG` | `1` for verbose tracing |
| `WORK_DIR` | Override working directory |
| `MAX_JOBS` | Parallel compilation jobs (default: 16) |
| `SERVER_PORT` | Server port (default: 8000) |

## Requirements

- Python 3.12+, [uv](https://docs.astral.sh/uv/), CUDA toolkit
