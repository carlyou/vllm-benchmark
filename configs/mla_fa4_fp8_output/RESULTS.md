# PR #43050 — MLA prefill FA4 fused-FP8 output: results

Model: `RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8` · GPU: 1×B200 (SM100) ·
feat branch `test--mla-fa4-native-fp8-output` vs `main`.

Raw logs: `~/.cache/vllm-bench/results/mla_fa4_fp8_output_bs1_swapped/` —
`bs1_bench_full.log` (E2E) and `microbench_final.log` (kernel microbench).

## 1. Kernel microbench (per-`forward_mha`, CUDA-graph timed, 50 repeats)

### 1a. Fused FP8 write vs bf16 + post-quant, fixed FA4

`benchmark.py --config configs/mla_fa4_fp8_output.yaml`. Same fa4 kernel; the
delta is the standalone post-quant kernel the fused write removes.

| Prefill spec | post_quant | fused | reduction |
| ------------ | ---------: | ----: | --------: |
| q512  |   37 µs |   33 µs | −11% |
| q1k   |   52 µs |   47 µs |  −9% |
| q2k   |   80 µs |   73 µs |  −9% |
| q4k   |  150 µs |  133 µs | −11% |
| q8k   |  396 µs |  368 µs |  −7% |
| 2×q4k |  283 µs |  252 µs | −11% |
| 4×q4k |  569 µs |  506 µs | −11% |
| 8×q4k | 1132 µs |  996 µs | −12% |

Consistent ~7–12% reduction in the prefill attention-write step.

### 1b. Prefill backend sweep, fixed bf16 + post-quant write

`benchmark.py --config configs/mla_fp8_output_backends.yaml`. flashinfer is
excluded (its prefill scans the model for MLAAttention layers, which the mock
harness does not provide).

| Prefill spec | fa4 | trtllm | trtllm vs fa4 |
| ------------ | --: | -----: | ------------: |
| q512  |   37 µs |   37 µs |  +0.5% |
| q1k   |   52 µs |   54 µs |  +4.1% |
| q2k   |   81 µs |   84 µs |  +4.8% |
| q4k   |  149 µs |  191 µs | +27.9% |
| q8k   |  395 µs |  460 µs | +16.5% |
| 2×q4k |  280 µs |  320 µs | +14.5% |
| 4×q4k |  570 µs |  610 µs |  +7.1% |
| 8×q4k | 1133 µs | 1183 µs |  +4.5% |

FA4 is fastest at every size; trtllm is 0.5–28% slower.

## 2. E2E bs1 / concurrency-1 TTFT — delta tracks run order, not branch

`vllm bench serve`, max-concurrency 1, input 4096, output 8, 100 prompts.
Run twice with swapped branch order:

| Order | runs 1st | runs 2nd |
| ----- | -------: | -------: |
| feat→main (original) | feat 180.5 | main 177.2 |
| main→feat (swapped)  | main 183.3 | feat 176.1 |

Mean TTFT (ms). The first-run branch is ~180–183 ms and the second is ~176–177 ms
regardless of which branch — a ~5–7 ms warmup penalty on whichever runs first.
main alone moved 177.2 → 183.3 just by running first. feat/main are at parity;
the earlier "feat slower" was feat running first. Saved post-quant kernel
(~0.1–0.7 ms/request over 27 layers) is below the ~180 ms TTFT noise floor.

Swapped-order full table:

| Metric | main_warmup | main_baseline | feat_warmup | feat_baseline |
| ------ | ----------: | ------------: | ----------: | ------------: |
| Benchmark duration (s) | 6.15 | 64.14 | 6.42 | 61.16 |
| Mean TTFT (ms)   | 180.19 | 183.27 | 182.67 | 176.11 |
| Median TTFT (ms) | 179.63 | 182.91 | 182.05 | 175.80 |
| P99 TTFT (ms)    | 190.09 | 189.54 | 194.57 | 195.43 |
| Mean TPOT (ms)   |  62.10 |  65.43 |  65.62 |  62.19 |

## 3. Correctness — fused FP8 output matches the post-quant path

`tests/v1/attention/test_mla_prefill_quant_output.py::test_fa4_fused_fp8_output_matches_post_quant`
(B200, feat venv): **PASSED** (plus the 17 existing gating tests). FA4's fused
e4m3 write matches bf16 attention + standalone static-FP8 quant within ≤1 e4m3
ULP, with ≥90% of elements in the exact same fp8 bucket.
