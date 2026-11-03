# 2026-10-31: PC Radeon RX 560 4 GB

One run of the standard benchmark. Results table and prompt are in the [README](../../README.md).

## System

| | |
|:--|:--|
| **Device** | PC Radeon RX 560 4 GB |
| **Chip / GPU** | Intel Core i3-8100 / AMD Radeon RX 560 4 GB (Polaris 11, Vulkan via RADV, Mesa 25.0.7) |
| **Memory for models** | 4 GB VRAM (4,096 MiB, no desktop, 4,091 MiB free at test start) |
| **Operating system** | Debian GNU/Linux 13 (trixie), kernel 6.12.101 |
| **Date** | 2026-10-31 |

## Tests

<a id="ollama-qwen3-5-4b-text-q4-k-m"></a>
### Ollama 0.40.13, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.13
- **Model (exact tag):** qwen3.5-4b-text:q4_K_M (from qwen3.5:4b-q4_K_M)
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 8,192
- **Cold load (s):** 2.22
- **TTFT cold (s):** 5.69
- **TTFT warm (s):** 60.68
- **TTFT with cache (s):** 4.39
- **Prompt processing tok/s:** 135
- **Generation tok/s:** 18.3
- **Generation tok/s at max context:** 18.3
- **Notes:** Custom model qwen3.5-4b-text:q4_K_M = qwen3.5:4b-q4_K_M without the vision projector (same GGUF blob sha256-a9d0a3e1, parameters unchanged). With the projector (0.68 GB), Ollama puts only 73% on the GPU on this 4 GB card, even at 4k context. Ollama service with OLLAMA_VULKAN=1 and OLLAMA_NUM_PARALLEL=1. Cold start, warm and cache with num_ctx 9,216 (at 16,384 only 94% GPU), 100% GPU, 2,988 of 4,096 MiB VRAM. Prompt cache: 7,699 of 8,210 tokens from the cache, TTFT 4.39 s instead of 60.7 s. Max context: 8k with 100% GPU, 2,921 MiB VRAM, prompt processing at 8k 137 tok/s; at 16k Ollama puts only 94% on the GPU, ladder ended. Previously loaded models were unloaded for the run. Environment: home server without restart (uptime 73 days) and without monitor, operated over SSH. Other services kept running in parallel, including game servers, Docker and databases; 7.7 GB RAM, 7.1 GB swap in use, stable during the measurement. Vulkan via RADV (Mesa 25.0.7); no desktop on the RX 560. Crash protection: reserve of 0.5 GB VRAM instead of 1.5 GB (4 GB card, Linux without TDR), plus abort at more than 500 MB growth in GTT (spill into system memory), the guard did not trigger. Input: German prose (17 Wikipedia articles), token count verified with the model tokenizer, raw text without chat template (no thinking). Median of 3 runs.

<a id="llama-cpp-qwen3-5-4b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12024, Vulkan), Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** llama.cpp 0.8.0-dev (build 12024, commit ddf48f85c, Vulkan)
- **Model (exact tag):** Ollama blob of qwen3.5:4b-q4_K_M (sha256-a9d0a3e1)
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 16,384
- **Cold load (s):** 1.53
- **TTFT cold (s):** 4.95
- **TTFT warm (s):** 60.41
- **TTFT with cache (s):** 4.33
- **Prompt processing tok/s:** 136
- **Generation tok/s:** 20.4
- **Generation tok/s at max context:** 18.9
- **Notes:** Official release llama-b12024-bin-ubuntu-vulkan-x64, without mmproj. `llama-server -ngl 99 -fa on --fit off -np 1`, context 16,384, 3,218 of 4,096 MiB VRAM. Load = process start until /health ok; TTFT cold = load + first request. Prompt cache: 7,695 of 8,210 tokens from the cache, TTFT 4.33 s instead of 60.5 s. Max context: 16k, 3,185 MiB VRAM, prompt processing at 16k 122 tok/s; 32k too large on paper (3,697 MiB + 512 MiB reserve > 4,096 MiB), not started. llama-bench (3 repetitions): pp512 155 tok/s, pp8192 136 tok/s, tg256 22.7 tok/s; at depth 15,360: pp512 103 tok/s, tg256 19.4 tok/s. Ollama had no model loaded during this phase (checked at each measurement). Environment: home server without restart (uptime 73 days) and without monitor, operated over SSH. Other services kept running in parallel, including game servers, Docker and databases; 7.7 GB RAM, 7.1 GB swap in use, stable during the measurement. Vulkan via RADV (Mesa 25.0.7); no desktop on the RX 560. Crash protection: reserve of 0.5 GB VRAM instead of 1.5 GB (4 GB card, Linux without TDR), plus abort at more than 500 MB growth in GTT (spill into system memory), the guard did not trigger. Input: German prose (17 Wikipedia articles), token count verified with the model tokenizer, raw text without chat template (no thinking). Median of 3 runs.

<a id="ollama-qwen3-14b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3 14B Q4_K_M (M)

- **Engine + version:** Ollama 0.40.13
- **Model (exact tag):** qwen3:14b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 9.3 GB model > 4 GB VRAM, not downloaded. Class skipped.

<a id="ollama-qwen3-8-27b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** Ollama 0.40.13
- **Model (exact tag):** qwen3.8:27b-q4_K_M
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 17.7 GB model > 4 GB VRAM, not downloaded. Class skipped.

<a id="llama-cpp-qwen3-14b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12024, Vulkan), Qwen3 14B Q4_K_M (M)

- **Engine + version:** llama.cpp 0.8.0-dev (build 12024, Vulkan)
- **Model (exact tag):** Qwen3 14B Q4_K_M (GGUF)
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 9.3 GB model > 4 GB VRAM, not downloaded. Class skipped.

<a id="llama-cpp-qwen3-8-27b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12024, Vulkan), Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** llama.cpp 0.8.0-dev (build 12024, Vulkan)
- **Model (exact tag):** Qwen3.8 27B Q4_K_M (GGUF)
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** does not fit
- **All measurements:** not measured
- **Notes:** Does not fit: about 17.7 GB model > 4 GB VRAM, not downloaded. Class skipped.

## Summary

The best setup on this device is llama.cpp 0.8.0-dev (build 12024, Vulkan) with Qwen3.5 4B Q4_K_M: 20.4 tok/s generation, 136 tok/s prompt processing and 16k context completely on the RX 560, about 11% faster generation than Ollama (18.3 tok/s) with twice the context. Notable is the slow prompt processing of the Polaris card (8,192 tokens take about 60 s, the prefix cache brings TTFT down to about 4.4 s), and Ollama only fits completely into the 4 GB without the vision projector. There were no crashes or guard triggers; Ollama stopped at 16k because of 94% GPU, 32k was ruled out on paper with llama.cpp, and classes M and L do not fit into 4 GB VRAM.
