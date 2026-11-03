# 2026-10-26: PC RTX 4060 Ti 8 GB

One run of the standard benchmark. Results table and prompt are in the [README](../../README.md).

## System

| | |
|:--|:--|
| **Device** | PC RTX 4060 Ti 8 GB |
| **Chip / GPU** | AMD Ryzen 9 3950X / NVIDIA GeForce RTX 4060 Ti (driver 610.88) |
| **Memory for models** | 8 GB VRAM (8,188 MiB, about 6.9 GB free with the desktop running) |
| **Operating system** | Windows 11 Pro N 25H2 (build 26200) |
| **Date** | 2026-10-26 |

## Tests

<a id="ollama-qwen3-5-4b-q4-k-m"></a>
### Ollama 0.40.11, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.11
- **Model (exact tag):** qwen3.5:4b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 65,536
- **Cold load (s):** 2.98
- **TTFT cold (s):** 3.19
- **TTFT warm (s):** 1.74
- **TTFT with cache (s):** 0.51
- **Prompt processing tok/s:** 4,938
- **Generation tok/s:** 59.4
- **Generation tok/s at max context:** 40.8
- **Notes:** Ollama default settings, num_ctx 16,384 or the step value, vision projector is loaded too. First cold start right after the download: load 41.9 s (file not in the file cache, not in the median). TTFT first cache request 2.42 s, second 0.51 s (prefix cache works). Max context: 64k with 100% GPU, peak 6,350 of 8,188 MiB; prompt processing at 64k 2,693 tok/s; 128k too large on paper (about 8.5 GB), not started. Model's max context: 262,144. Crash protection: CUDA Sysmem Fallback Policy = Prefer No Sysmem Fallback; TDR values at Windows default (2 s, not changed); no iGPU (Ryzen 9 3950X), monitor on the RTX 4060 Ti; desktop uses about 0.6 to 1.1 GB VRAM. No GPU driver reset. Input: German prose (Fontane, Project Gutenberg), token count verified with the model tokenizer, raw text without chat template (no thinking). Cold start, warm and cache with context 16,384; median of 3 runs.

<a id="llama-cpp-qwen3-5-4b-q4-k-m"></a>
### llama.cpp b11889 (CUDA 13.4), Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** llama.cpp b11889 (CUDA 13.4)
- **Model (exact tag):** unsloth/Qwen3.5-4B-GGUF: Qwen3.5-4B-Q4_K_M.gguf
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 65,536
- **Cold load (s):** 1.86
- **TTFT cold (s):** 2.01
- **TTFT warm (s):** 1.56
- **TTFT with cache (s):** 0.20
- **Prompt processing tok/s:** 5,420
- **Generation tok/s:** 72.2
- **Generation tok/s at max context:** 48.1
- **Notes:** GGUF from unsloth (Qwen publishes no GGUF of its own for 4B), without mmproj. `llama-server -ngl 99 --fit off -np 1`, flash attention auto, ubatch 512. Load = process start until /health ok; TTFT cold = load + first request (0.15 s). Prompt cache: 7,695 of 8,210 tokens from the cache, TTFT 0.20 s instead of 2.23 s. Max context: 64k, peak 5,568 of 8,188 MiB; prompt processing at 64k 2,844 tok/s; 128k too large on paper (about 7.7 GB), not started. llama-bench (3 repetitions): pp8192 5,525 tok/s, tg256 80.8 tok/s; at depth 64,512: pp512 1,292 tok/s, tg256 48.6 tok/s. Crash protection: CUDA Sysmem Fallback Policy = Prefer No Sysmem Fallback; TDR values at Windows default (2 s, not changed); no iGPU (Ryzen 9 3950X), monitor on the RTX 4060 Ti; desktop uses about 0.6 to 1.1 GB VRAM. No GPU driver reset. Input: German prose (Fontane, Project Gutenberg), token count verified with the model tokenizer, raw text without chat template (no thinking). Cold start, warm and cache with context 16,384; median of 3 runs.

<a id="ollama-qwen3-14b-q4-k-m"></a>
### Ollama 0.40.11, Qwen3 14B Q4_K_M (M)

- **Engine + version:** Ollama 0.40.11
- **Model (exact tag):** qwen3:14b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: 9.3 GB model > 6.9 GB free GPU memory. Qwen3.x has no model around 14B, hence Qwen3 14B; qwen3.5:9b (6.6 GB) does not fit either with KV cache and 1.5 GB reserve. Class skipped.

<a id="ollama-qwen3-8-27b-q4-k-m"></a>
### Ollama 0.40.11, Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** Ollama 0.40.11
- **Model (exact tag):** qwen3.8:27b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: 17.7 GB model > 6.9 GB free GPU memory. Class skipped.

<a id="llama-cpp-qwen3-14b-q4-k-m"></a>
### llama.cpp b11889 (CUDA 13.4), Qwen3 14B Q4_K_M (M)

- **Engine + version:** llama.cpp b11889 (CUDA 13.4)
- **Model (exact tag):** Qwen3-14B Q4_K_M (GGUF)
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 9 GB model > 6.9 GB free GPU memory, not downloaded. Class skipped.

<a id="llama-cpp-qwen3-8-27b-q4-k-m"></a>
### llama.cpp b11889 (CUDA 13.4), Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** llama.cpp b11889 (CUDA 13.4)
- **Model (exact tag):** Qwen3.8 27B Q4_K_M (GGUF)
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 17 GB model > 6.9 GB free GPU memory, not downloaded. Class skipped.

## Summary

The best setup on this device is llama.cpp b11889 (CUDA 13.4) with Qwen3.5 4B Q4_K_M: 72.2 tok/s generation, 5,420 tok/s prompt processing and 64k context completely on the GPU, about 20% faster generation than Ollama (59.4 tok/s). Notable are the large gain from the prefix cache (TTFT 0.20 s or 0.51 s instead of about 2.3 s) and the clear drop at a full 64k context (generation down 33% and 31%, prompt processing about halved). There were no aborts or driver resets; classes M (14B) and L (Qwen3.8 27B) do not fit into 8 GB of GPU memory and were skipped, and 128k context was ruled out on paper.
