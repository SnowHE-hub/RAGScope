# 4K / 8K fixed-context resource diagnosis

COMPLETE: 16 formal requests and 6 separate warmups, all real Qwen3.5-2B Q4_K_M requests; zero request/output failures. Both windows were unloaded after completion. These eight source-feasible dev questions are a resource diagnosis and are excluded from the main quality ranking. All92 excluded questions retain their original reasons in the frozen manifest.

| Window | Actual input tokens | Formal / warmup | Mean / P95 generation (s) | Sampled device-total maximum (bytes) | Minimum sampled host available (bytes) |
|---|---:|---:|---:|---:|---:|
| 4096 | 3584 | 8 / 3 | 1.366 / 1.648 | 1517289472 | 2847301632 |
| 8192 | 7680 | 8 / 3 | 1.920 / 2.319 | 1628438528 | 2818744320 |

Each request used the same frozen prompt/schema/greedy512 protocol and exact original canonical spans. All22 actual backend prompt counts match their frozen token counters. P95 uses linear interpolation over the eight formal observations; warmup is excluded. generation_ms includes local telemetry finalization. Device usage is whole-device sampled total, not this process peak; service RSS can double-count shared pages, and sampling can miss peaks. Actual layer offload count and first-token time are unavailable.

The R3 reranker and auxiliary embedding worker/server were concurrently active with configured CPU4. Their exact PID/create/argv and fresh resource measurements are retained in EXECUTION_CONDITIONS_20261014T0322.json and PORTABLE_PREFLIGHT_20261014T0330.json. Consequently these latency measurements are not an isolated model benchmark.

System Ollama had automatically changed to0.40.3. The empty own service was stopped before any model request. The [official v0.34.4 portable release](https://github.com/ollama/ollama/releases/tag/v0.34.4) was downloaded once to D, checked against its published SHA535193f38f3344e5b08f5d1c171c31ce11aa17f0124ff69ae26d8ec7fe06fa62, and unpacked with all82 file hashes. Its executable SHA d571e4cf0ad4456fccd33d5b985b6d63ba5669c6e5a812391704083952aae3d5 equals the historical pilot/E3 executable. The project service now accepts an explicit --binary path, preserving the portable library tree. System files, original11434, drivers, frozen experiment identity and stress sources were not changed. Twenty-nine CPU startup/lease contracts passed.

After the final unload, API ps was empty, GPU usage12MiB/0%, host available3,870,851,072bytes. Owned11435 remains READY with no resident model while waiting for sealed E2 R3 TEST200 input; this is not E2 or main-matrix completion. Global budget is1443/10000, core1200, and tickets1422–1443 account for exactly22 new calls.

Completion receipt: [COMPLETION_RECEIPT.json](D:/RAGScope_Runtime/runs/20261014_v2_context_stress/COMPLETION_RECEIPT.json), SHA `08c23b7079f2e720fad240adf8f7ce78dda8c19553fe6d93c6617f0f5ad451bf`.
Release evidence: [RELEASE_ALL_STRESS.json](D:/RAGScope_Runtime/runs/20261014_v2_context_stress/RELEASE_ALL_STRESS.json).

Historical RAM and backend-version guards remain preserved separately from model failures. No additional combinations or model requests were introduced.
