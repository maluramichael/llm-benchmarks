# 2026-10-26: MacBook Pro M5 Max 128 GB

One run of the standard benchmark. Results table and prompt are in the [README](../../README.md). This run is measured in stages whenever the Mac is free. Finished so far: Ollama with Qwen3.5 4B. Still to come: Ollama M, L and XL, llama.cpp, llama-bench and MLX.

## System

| | |
|:--|:--|
| **Device** | MacBook Pro M5 Max 128 GB |
| **Chip / GPU** | Apple M5 Max (Mac17,6), 18 CPU cores (6 Super, 12 Performance), 40 GPU cores, 128 GB unified memory |
| **Memory for models** | 107.5 GB (Metal recommendedMaxWorkingSetSize 110,100 MiB, 84% of unified memory, iogpu.wired_limit_mb unchanged) |
| **Operating system** | macOS 26.6 (build 25G72) |
| **Date** | 2026-10-26 (Ollama S measured on 2026-10-26 and 2026-10-27) |

## Tests

<a id="ollama-qwen3-5-4b-q4-k-m"></a>
### Ollama 0.40.11, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.11 (macOS app, Metal)
- **Model (exact tag):** qwen3.5:4b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 262,144 (model maximum, completely on the GPU, 13.0 GB in use)
- **Cold load (s):** 0.82
- **TTFT cold (s):** 0.98
- **TTFT warm (s):** 1.99
- **TTFT with cache (s):** 0.31
- **Prompt processing tok/s:** 4,171
- **Generation tok/s:** 118.0
- **Generation tok/s at max context:** 36.2 (prompt processing 848)
- **Notes:** Standard prompt, median of 3 runs, raw text completion, base context 16k, one slot (OLLAMA_NUM_PARALLEL 1). On the Mac the plain tag qwen3.5:4b resolves to an MLX variant, hence the explicit tag with -q4_K_M (same GGUF file as on the PC RTX 3090 24 GB). Measured after a fresh restart with Colima, browsers and other apps closed, only the built-in display on, on the power supply (macOS energy mode "High Power", on battery the speed drops to about a third). Context steps (prompt processing / generation tok/s): 8k 4,105 / 117.7, 16k 3,654 / 109.7, 32k 3,018 / 95.0, 64k 2,226 / 75.1, 128k 1,474 / 56.8, 256k 848 / 36.2, swap stayed at 0. In the prompt cache test, 7,187 of 8,210 tokens came from the cache.

## Summary

Only Qwen3.5 4B with Ollama 0.40.11 has been measured so far: 4,171 tok/s prompt processing and 118 tok/s generation, and still 848 and 36 tok/s at the full 262,144 tokens of context. Notably, the Mac is slower than the RTX 3090 on the small model (6,626 and 150 tok/s), but handles the full context without swap. There were no aborts, the remaining tests follow in further stages.
