#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Reusable A/B benchmark script for vLLM branches.
#
# Compares any number of (branch, config) pairs defined in a YAML config file.
# Handles per-branch cloning, venv setup, build caching, server lifecycle,
# and result collection.
#
# Usage:
#   ./benchmark_ab.sh configs/mla_fusion.yaml
#   CUDA_ARCH=12.1 ./benchmark_ab.sh configs/mla_fusion.yaml
#   DEBUG=1 FORCE_BUILD=1 ./benchmark_ab.sh configs/mla_fusion.yaml
#
# Machine-specific env vars (override YAML):
#   CUDA_ARCH      - e.g. "12.1" for GB10; sets CMAKE_CUDA_ARCHITECTURES
#   VLLM_USE_PRECOMPILED - override use_precompiled from YAML (0 or 1)
#   DEBUG          - set to 1 for bash -x tracing
#   FORCE_BUILD    - set to 1 to rebuild even if .build_done matches HEAD
#   TORCH_INDEX    - PyTorch wheel index URL
#   WORK_DIR       - override work_dir from YAML
#   MAX_JOBS       - parallel compilation jobs (default: 16)
#   SERVER_PORT    - port for vllm server (default: 8000)
#   SERVER_WAIT_TIMEOUT - seconds to wait for server health (default: 1800)
#
# Build env vars (passed through to source builds):
#   VLLM_FLASH_ATTN_SRC_DIR - local flash-attention source dir
#   VLLM_CUTLASS_SRC_DIR    - local CUTLASS source dir
#   VLLM_TARGET_DEVICE      - target device (default: cuda)
#   CMAKE_BUILD_TYPE        - Debug/Release/RelWithDebInfo
#   CMAKE_ARGS              - extra cmake args

set -euo pipefail
[[ "${DEBUG:-}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${1:-}"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
CUDA_ARCH="${CUDA_ARCH:-}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-1800}"

# ── Parse YAML config ─────────────────────────────────────────────────
# Uses a Python one-liner to extract fields. Outputs shell variable assignments.
parse_config() {
    local yaml_file=$1
    python3 -c "
import yaml, json, sys, os, shlex

with open('$yaml_file') as f:
    cfg = yaml.safe_load(f)

def q(v):
    return shlex.quote(str(v))

# Top-level fields with defaults
print(f'CFG_REPO={q(cfg.get(\"repo\", \"https://github.com/vllm-project/vllm.git\"))}')
print(f'CFG_MODEL={q(cfg.get(\"model\", \"meta-llama/Llama-3.1-8B-Instruct\"))}')
print(f'CFG_TP={q(cfg.get(\"tp\", 1))}')
print(f'CFG_MAX_MODEL_LEN={q(cfg.get(\"max_model_len\", 4096))}')
print(f'CFG_NUM_PROMPTS={q(cfg.get(\"num_prompts\", 200))}')
print(f'CFG_INPUT_LEN={q(cfg.get(\"input_len\", 128))}')
print(f'CFG_OUTPUT_LEN={q(cfg.get(\"output_len\", 128))}')
print(f'CFG_USE_PRECOMPILED={q(int(cfg.get(\"use_precompiled\", True)))}')
gpu_mem = cfg.get('gpu_memory_utilization')
if gpu_mem is not None:
    print(f'CFG_GPU_MEMORY_UTILIZATION={q(gpu_mem)}')

work_dir = cfg.get('work_dir', '~/vllm_ab_benchmark')
work_dir = os.path.expanduser(work_dir)
print(f'CFG_WORK_DIR={q(work_dir)}')

runs = cfg.get('runs', [])
print(f'CFG_NUM_RUNS={q(len(runs))}')

for i, run in enumerate(runs):
    print(f'CFG_RUN_{i}_LABEL={q(run[\"label\"])}')
    print(f'CFG_RUN_{i}_BRANCH={q(run[\"branch\"])}')
    print(f'CFG_RUN_{i}_COMMIT={q(run.get(\"commit\", \"\"))}')
    cc = run.get('compilation_config')
    if cc:
        print(f'CFG_RUN_{i}_CC={q(json.dumps(cc))}')
    else:
        print(f\"CFG_RUN_{i}_CC=''\")

# Deduplicated (branch, commit) install units
seen = {}
installs = []
for run in runs:
    key = (run['branch'], run.get('commit', ''))
    if key not in seen:
        seen[key] = len(installs)
        installs.append(key)
print(f'CFG_NUM_INSTALLS={q(len(installs))}')
for i, (b, c) in enumerate(installs):
    print(f'CFG_INSTALL_{i}_BRANCH={q(b)}')
    print(f'CFG_INSTALL_{i}_COMMIT={q(c)}')
"
}

# ── Load config ───────────────────────────────────────────────────────
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
eval "$(parse_config "$CONFIG_FILE")"

# Env vars override YAML values
CFG_WORK_DIR="${WORK_DIR:-$CFG_WORK_DIR}"
CFG_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-$CFG_USE_PRECOMPILED}"
REPOS_DIR="$CFG_WORK_DIR/repos"
RESULTS_DIR="$CFG_WORK_DIR/results"
LOGS_DIR="$CFG_WORK_DIR/logs"

mkdir -p "$REPOS_DIR" "$RESULTS_DIR" "$LOGS_DIR"

echo "============================================"
echo "  vLLM A/B Benchmark"
echo "============================================"
echo "Config:      $CONFIG_FILE"
echo "Model:       $CFG_MODEL"
echo "TP:          $CFG_TP"
echo "Max len:     $CFG_MAX_MODEL_LEN"
echo "Prompts:     $CFG_NUM_PROMPTS"
echo "Input len:   $CFG_INPUT_LEN"
echo "Output len:  $CFG_OUTPUT_LEN"
echo "Precompiled: $CFG_USE_PRECOMPILED"
[[ -n "$CUDA_ARCH" ]] && echo "CUDA arch:   $CUDA_ARCH"
echo "Work dir:    $CFG_WORK_DIR"
echo "Runs:        $CFG_NUM_RUNS"
echo "============================================"
echo ""

# ── Helper: wait for server to be ready ───────────────────────────────
wait_for_server() {
    echo "Waiting for server on port $SERVER_PORT (timeout ${SERVER_WAIT_TIMEOUT}s)..."
    local elapsed=0
    while (( elapsed < SERVER_WAIT_TIMEOUT )); do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "ERROR: Server did not start within ${SERVER_WAIT_TIMEOUT}s"
    return 1
}

# ── Helper: kill server if running ────────────────────────────────────
kill_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "Stopping server (PID=$SERVER_PID)..."
        # Kill entire process group (server + engine workers)
        kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
        sleep 3
    fi
}
trap kill_server EXIT

# ── Helper: sanitize branch+commit for directory use ─────────────────
branch_to_dir() {
    local branch=$1
    local commit=${2:-}
    local dir
    dir=$(echo "$branch" | sed 's|/|--|g')
    if [[ -n "$commit" ]]; then
        dir="${dir}--${commit:0:8}"
    fi
    echo "$dir"
}

# ── Install a branch ─────────────────────────────────────────────────
install_branch() {
    local branch=$1
    local commit=${2:-}
    local dir_name
    dir_name=$(branch_to_dir "$branch" "$commit")
    local repo_dir="$REPOS_DIR/$dir_name"

    echo ""
    echo "============================================"
    echo "  Installing: $branch${commit:+ @ $commit}"
    echo "============================================"

    # Clone or update
    if [[ ! -d "$repo_dir" ]]; then
        git clone "$CFG_REPO" "$repo_dir"
    fi

    (
        cd "$repo_dir"
        git fetch origin "$branch"
        if [[ -n "$commit" ]]; then
            git checkout "$commit"
        else
            git checkout "$branch"
            git reset --hard "origin/$branch"
        fi

        # Create per-repo venv if needed
        if [[ ! -d ".venv" ]]; then
            echo "Creating venv..."
            uv venv --python 3.12 --seed
        fi
        source .venv/bin/activate

        # Check .build_done marker
        local current_head
        current_head=$(git rev-parse HEAD)
        if [[ "$FORCE_BUILD" != "1" ]] && [[ -f ".build_done" ]]; then
            local built_hash
            built_hash=$(cat .build_done)
            if [[ "$built_hash" == "$current_head" ]]; then
                echo "Build already done for $current_head — skipping."
                return 0
            fi
        fi

        # Install torch + build deps (needed for both paths with --no-build-isolation)
        uv pip install torch --extra-index-url "$TORCH_INDEX"
        grep -v '^torch==' requirements/build.txt | uv pip install -r -

        if [[ "$CFG_USE_PRECOMPILED" == "1" ]]; then
            # Fast path: download precompiled CUDA kernels
            echo "Installing vllm (precompiled, HEAD=$current_head)..."
            export VLLM_USE_PRECOMPILED=1
            uv pip install -e . --no-build-isolation
        else
            # Source build: compile CUDA kernels from source
            echo "Building vllm from source (HEAD=$current_head)..."

            # Pass through build env vars
            export MAX_JOBS="${MAX_JOBS:-16}"
            [[ -n "${VLLM_FLASH_ATTN_SRC_DIR:-}" ]] && export VLLM_FLASH_ATTN_SRC_DIR
            [[ -n "${VLLM_CUTLASS_SRC_DIR:-}" ]] && export VLLM_CUTLASS_SRC_DIR
            [[ -n "${VLLM_TARGET_DEVICE:-}" ]] && export VLLM_TARGET_DEVICE
            [[ -n "${CMAKE_BUILD_TYPE:-}" ]] && export CMAKE_BUILD_TYPE

            if [[ -n "$CUDA_ARCH" ]]; then
                export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH"
                # PyTorch wheels bake in a default CMAKE_CUDA_ARCHITECTURES
                # (e.g. 75) that overrides TORCH_CUDA_ARCH_LIST, so we must
                # pass the correct value explicitly via CMAKE_ARGS.
                local cmake_arch
                cmake_arch=$(echo "$CUDA_ARCH" | tr -d '.')
                export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_CUDA_ARCHITECTURES=${cmake_arch}"
            fi

            uv pip install -e . --no-build-isolation
        fi

        # Write marker
        echo "$current_head" > .build_done
        echo "Build complete."
    )
}

# ── Run one benchmark ────────────────────────────────────────────────
run_benchmark() {
    local label=$1
    local branch=$2
    local commit=$3
    local cc_json=$4  # compilation_config JSON string, or empty

    local dir_name
    dir_name=$(branch_to_dir "$branch" "$commit")
    local repo_dir="$REPOS_DIR/$dir_name"
    local outfile="$RESULTS_DIR/${label}.txt"

    echo ""
    echo "----------------------------------------------"
    echo "  Run: $label"
    echo "  Branch: $branch${commit:+ @ $commit}"
    [[ -n "$cc_json" ]] && echo "  Compilation config: $cc_json"
    echo "----------------------------------------------"

    # Build server command args (only universally-supported flags)
    local server_args=(
        --model "$CFG_MODEL"
        --tensor-parallel-size "$CFG_TP"
        --max-model-len "$CFG_MAX_MODEL_LEN"
        --trust-remote-code
        --port "$SERVER_PORT"
    )
    if [[ -n "${CFG_GPU_MEMORY_UTILIZATION:-}" ]]; then
        server_args+=(--gpu-memory-utilization "$CFG_GPU_MEMORY_UTILIZATION")
    fi
    if [[ -n "$cc_json" ]]; then
        server_args+=(-cc "$cc_json")
    fi

    # Start server in its own process group so kill_server can reap all children
    setsid bash -c "
        cd '$repo_dir'
        source .venv/bin/activate
        python -m vllm.entrypoints.openai.api_server $(printf '%q ' "${server_args[@]}") \
            > '$LOGS_DIR/${label}_server.log' 2>&1
    " &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "Server log tail:"
        tail -30 "$LOGS_DIR/${label}_server.log"
        kill_server
        return 1
    fi

    # Sanity check
    echo "Sanity check: generating one response..."
    local sanity_file="$LOGS_DIR/${label}_sanity.json"
    curl -s "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$CFG_MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
            \"max_tokens\": 32
        }" > "$sanity_file"

    if python3 -c "
import json, sys
with open('$sanity_file') as f:
    d = json.load(f)
print('Response:', d['choices'][0]['message']['content'][:80])
" 2>/dev/null; then
        echo "Sanity check passed."
    else
        echo "WARNING: Sanity check failed. Response:"
        head -5 "$sanity_file"
    fi

    # Run benchmark
    echo "Running vllm bench serve ($CFG_NUM_PROMPTS prompts)..."
    (
        cd "$repo_dir"
        source .venv/bin/activate
        vllm bench serve \
            --backend vllm \
            --model "$CFG_MODEL" \
            --port "$SERVER_PORT" \
            --num-prompts "$CFG_NUM_PROMPTS" \
            --request-rate inf \
            --random-input-len "$CFG_INPUT_LEN" \
            --random-output-len "$CFG_OUTPUT_LEN" \
            --ignore-eos \
            2>&1 | tee "$outfile"
    )

    kill_server
    echo "Results saved to $outfile"
}

# ── Main: install branches ────────────────────────────────────────────
echo "Installing $CFG_NUM_INSTALLS unique branch(es)..."
for (( i=0; i<CFG_NUM_INSTALLS; i++ )); do
    branch_var="CFG_INSTALL_${i}_BRANCH"
    commit_var="CFG_INSTALL_${i}_COMMIT"
    install_branch "${!branch_var}" "${!commit_var}"
done

# ── Main: run benchmarks ─────────────────────────────────────────────
echo ""
echo "Running $CFG_NUM_RUNS benchmark(s)..."
for (( i=0; i<CFG_NUM_RUNS; i++ )); do
    label_var="CFG_RUN_${i}_LABEL"
    branch_var="CFG_RUN_${i}_BRANCH"
    commit_var="CFG_RUN_${i}_COMMIT"
    cc_var="CFG_RUN_${i}_CC"
    run_benchmark "${!label_var}" "${!branch_var}" "${!commit_var}" "${!cc_var}"
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""

printf "%-30s  %12s  %12s  %12s  %12s\n" \
    "LABEL" "THROUGHPUT" "MED_TTFT" "MED_TPOT" "P99_TPOT"
printf "%-30s  %12s  %12s  %12s  %12s\n" \
    "-----" "----------" "--------" "--------" "--------"

for f in "$RESULTS_DIR"/*.txt; do
    [[ -f "$f" ]] || continue
    label=$(basename "$f" .txt)

    throughput=$(grep -oP 'Request throughput.*?:\s*\K[\d.]+' "$f" 2>/dev/null || echo "N/A")
    median_ttft=$(grep -oP 'Median TTFT.*?:\s*\K[\d.]+' "$f" 2>/dev/null || echo "N/A")
    median_tpot=$(grep -oP 'Median TPOT.*?:\s*\K[\d.]+' "$f" 2>/dev/null || echo "N/A")
    p99_tpot=$(grep -oP 'P99 TPOT.*?:\s*\K[\d.]+' "$f" 2>/dev/null || echo "N/A")

    printf "%-30s  %10s r/s  %9s ms  %9s ms  %9s ms\n" \
        "$label" "$throughput" "$median_ttft" "$median_tpot" "$p99_tpot"
done

echo ""
echo "Full results in: $RESULTS_DIR/"
echo "Server logs in:  $LOGS_DIR/"
