# vLLM A/B Benchmark

Reusable harness for comparing vLLM branches side-by-side. Define runs in YAML — the tool handles cloning, building, server lifecycle, and result collection.

## Usage

```bash
# Full pipeline: build + benchmark
vllm-bench run configs/mla_quant_fusion/b200_fp8.yaml

# Individual steps
vllm-bench build configs/mla_quant_fusion/b200_fp8.yaml
vllm-bench bench configs/mla_quant_fusion/b200_fp8.yaml

# Filter to specific runs
vllm-bench run configs/mla_quant_fusion/b200_fp8.yaml --run main_baseline --run feature_fuse_on

# Override build/server settings
vllm-bench run configs/my_test.yaml --port 9000 --max-jobs 8

# Clean caches (flashinfer, torch compile, triton, venvs)
vllm-bench clean configs/my_test.yaml
vllm-bench clean --all  # also remove huggingface model cache
```

Force rebuild via environment variable:

```bash
FORCE_BUILD=1 vllm-bench run configs/my_test.yaml
```

## Config

Runs are organized under `branches`. Build config is set at the global or branch level. Server and bench config can be overridden at global, branch, or run level.

```yaml
project:
  repo: https://github.com/vllm-project/vllm.git
  model: RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8

build:
  max_jobs: 0.8              # <=1: fraction of CPU cores, >1: absolute

server:
  tp: 1
  max_model_len: 4096

bench:
  num_prompts: 1000
  input_len: 128
  output_len: 128
  request_rate: 50

branches:
  main:
    runs:
      - label: main_warmup
      - label: main_baseline

  my-feature-branch:
    build:
      use_precompiled: false   # branch-level build override
    runs:
      - label: feature_off
        server:                # per-run server override
          compilation_config:
            pass_config:
              fuse_attn_quant: false

      - label: feature_on
        server:
          compilation_config:
            pass_config:
              fuse_attn_quant: true
```

### Config hierarchy

| Level | build | server | bench | test | eval |
|---|---|---|---|---|---|
| Global (top-level) | yes | yes | yes | yes | yes |
| Branch (`branches.<name>`) | yes | yes | yes | yes | yes |
| Run (`branches.<name>.runs[]`) | no | yes | yes | yes | yes |

Effective config is merged: global -> branch -> run (for server/bench) or global -> branch (for build).

### Build options

| Field | Default | Description |
|---|---|---|
| `use_precompiled` | `true` | Use precompiled vllm wheel (main branch only) |
| `cuda_arch` | auto | CUDA architecture (e.g. `"9.0"`, `"12.1"`) |
| `max_jobs` | `1.0` | Build parallelism (<=1: fraction of cores) |
| `install_flash_attn` | `false` | Install flash-attn from source |
| `torch_index` | `cu130` | PyTorch wheel index URL |

### Server options

| Field | Default | Description |
|---|---|---|
| `tp` | `1` | Tensor parallel size |
| `max_model_len` | `4096` | Maximum sequence length |
| `enforce_eager` | `false` | Skip CUDA graph capture |
| `gpu_memory_utilization` | none | GPU memory fraction |
| `port` | `8000` | Server port |
| `wait_timeout` | `600` | Seconds to wait for server startup |
| `compilation_config` | none | vllm compilation config (JSON) |

### Bench options

| Field | Default | Description |
|---|---|---|
| `num_prompts` | `1000` | Number of benchmark requests |
| `input_len` | `128` | Random input length |
| `output_len` | `128` | Random output length |
| `request_rate` | `inf` | Requests per second |
| `warmup_prompts` | `3` | Warmup requests before benchmark |

### Test options

| Field | Default | Description |
|---|---|---|
| `script` | none | Pytest target relative to repo root (e.g. `tests/models/test_mla.py`) |
| `args` | none | Additional pytest CLI args (e.g. `-x -v --timeout=600`) |

Tests run directly in the built venv — no server is started.

## Subcommands

| Command | Description |
|---|---|
| `run` | Build + benchmark (full pipeline) |
| `build` | Clone and build all branches |
| `test` | Run pytest tests (builds must already exist) |
| `bench` | Benchmark only (builds must already exist) |
| `compile` | Pre-compile CUDA graphs (start/check/stop) |
| `clean` | Remove caches (flashinfer, torch compile, triton, venvs) |

## Build caching

Builds are cached per branch and skipped when the commit and build config haven't changed. Sequential execution allows later builds to reuse uv's package cache from earlier builds.

After install, the tool verifies that torch's CUDA version matches the system CUDA version. Precompiled vllm wheels may resolve torch with CUDA 12 on CUDA 13 systems, causing cuBLAS initialization failures on SM 100+ GPUs.

## Environment Variables

| Variable | Description |
|---|---|
| `FORCE_BUILD` | `1` to rebuild even if cached |

## Requirements

- Python 3.12+, [uv](https://docs.astral.sh/uv/), CUDA toolkit
