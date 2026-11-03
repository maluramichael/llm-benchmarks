# 2026-11-02: PC RTX 4090 24 GB

One run of the standard benchmark. Results table and prompt are in the [README](../../README.md).

## System

| | |
|:--|:--|
| **Device** | PC RTX 4090 24 GB |
| **Chip / GPU** | NVIDIA GeForce RTX 4090 (driver 621.01, CUDA 13.5), AMD Ryzen 7 9800X3D, 32 GB RAM |
| **Memory for models** | 24 GB VRAM (about 0.9 GB used by the desktop) |
| **Operating system** | Windows 11 IoT Enterprise LTSC (build 26300) |
| **Date** | 2026-11-02 |

Preparation: freshly restarted (2 minutes uptime at the start), tray and background apps quit (browser, fan control UI, package updater, peripheral tools), "CUDA - Sysmem Fallback Policy" set to "Prefer No Sysmem Fallback", TdrDelay 10 and TdrDdiDelay 20. One monitor stayed connected to the RTX 4090 (the CPU's iGPU is not active). The download of Qwen3.8 27B was still running in the background during the llama.cpp tests of Qwen3.5 4B and Qwen3 14B. Input: 17 German Wikipedia articles (1.89 MB), cut with each model's tokenizer; all requests are raw text completions, temperature 0, seed 42, 256 output tokens, median of 3 runs. No GPU driver reset during the whole run.

## Tests

<a id="llama-cpp-qwen3-5-4b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12034), Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** llama.cpp 0.8.0-dev, build 12034 (commit 1c5ed702c), llama-server, official release win-cuda-13.4-x64
- **Model (exact tag):** qwen3.5:4b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 262,144 (model maximum, completely on the GPU)
- **Cold load (s):** 1.93
- **TTFT cold (s):** 2.04
- **TTFT warm (s):** 0.61
- **TTFT with cache (s):** 0.07
- **Prompt processing tok/s:** 15,783
- **Generation tok/s:** 203.1
- **Generation tok/s at max context:** 72.5 (prompt processing 6,790)
- **Notes:** llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k. In the prompt cache test, 7,691 tokens came from the cache and 514 were processed again. Uses 4.5 GB VRAM at 16k and 12.4 GB at 256k (including the desktop). Answers often ended before 256 tokens (144 to 205), so generation is measured over fewer tokens. The 256k step ran at the very end of the run, so a possible GPU reset (as on the RTX 3090 with build 11909) could not cost other tests; the first attempt aborted because of a parsing error in the measuring script (SSE keep-alive line), the repeat completed without problems. llama-bench (`-p 512,8192 -n 256 -r 3`): prompt processing 17,613 tok/s at 512 and 17,258 tok/s at 8,192 tokens, generation 230.8 tok/s, at a depth of 130,048 tokens prompt processing 6,405 and generation 112.1 tok/s.

<a id="ollama-qwen3-5-4b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.13 (winget)
- **Model (exact tag):** qwen3.5:4b
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 262,144 (model maximum, 100% on the GPU according to ollama ps)
- **Cold load (s):** 1.78
- **TTFT cold (s):** 1.91
- **TTFT warm (s):** 0.51
- **TTFT with cache (s):** 0.11
- **Prompt processing tok/s:** 18,310
- **Generation tok/s:** 185.8
- **Generation tok/s at max context:** 69.7 (prompt processing 6,778)
- **Notes:** Own `ollama serve` with one slot (OLLAMA_NUM_PARALLEL=1) and f16 KV cache, base context 16k. Ollama starts its bundled llama-server with flash attention auto, batch 512 and the vision projector loaded. The first load after installation took 23.9 s (TTFT 44.9 s), presumably compiling CUDA kernels, then 1.8 s. Uses 5.4 GB VRAM at 16k and 13.2 GB at 256k. Shared GPU memory rose to at most 1.2 GB at 128k, by less than 500 MB per step. Answers ended before 256 tokens (155 to 213). The 256k step ran at the end of the run, like with llama.cpp.

<a id="llama-cpp-qwen3-14b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12034), Qwen3 14B Q4_K_M (M)

- **Engine + version:** llama.cpp 0.8.0-dev, build 12034 (commit 1c5ed702c), llama-server, official release win-cuda-13.4-x64
- **Model (exact tag):** qwen3:14b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** 40,960 (model maximum, completely on the GPU)
- **Cold load (s):** 5.35
- **TTFT cold (s):** 5.49
- **TTFT warm (s):** 1.61
- **TTFT with cache (s):** 0.03
- **Prompt processing tok/s:** 6,140
- **Generation tok/s:** 81.0
- **Generation tok/s at max context:** 55.7 (prompt processing 3,701)
- **Notes:** Class M uses Qwen3 instead of Qwen3.x, there is no Qwen3.x model around 14B in the Ollama library. llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k. In the prompt cache test, 8,193 tokens came from the cache and only 15 were processed again. Uses 12.4 GB VRAM at 16k and 16.3 GB at 40,960. llama-bench (`-p 512,8192 -n 256 -r 3`): prompt processing 6,971 tok/s at 512 and 6,193 tok/s at 8,192 tokens, generation 94.5 tok/s, at a depth of 39,936 tokens prompt processing 2,497 and generation 56.6 tok/s.

<a id="ollama-qwen3-14b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3 14B Q4_K_M (M)

- **Engine + version:** Ollama 0.40.13 (winget)
- **Model (exact tag):** qwen3:14b
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** 40,960 (model maximum, 100% on the GPU according to ollama ps)
- **Cold load (s):** 2.26
- **TTFT cold (s):** 2.44
- **TTFT warm (s):** 1.53
- **TTFT with cache (s):** 0.05
- **Prompt processing tok/s:** 6,293
- **Generation tok/s:** 81.3
- **Generation tok/s at max context:** 56.0 (prompt processing 3,768)
- **Notes:** Class M uses Qwen3 instead of Qwen3.x. Own `ollama serve` with one slot and f16 KV cache, base context 16k. The first load took 10.3 s, then 2.3 s. The first warm request generated only 35.4 tok/s (the other two 81.3 and 81.4), and the first prompt cache pair got no cache hit (TTFT 1.56 s, the other two 0.05 and 0.04 s); the medians are not affected. Uses 12.6 GB VRAM at 16k and 16.5 GB at 40,960.

<a id="llama-cpp-qwen3-8-27b-q4-k-m"></a>
### llama.cpp 0.8.0-dev (build 12034), Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** llama.cpp 0.8.0-dev, build 12034 (commit 1c5ed702c), llama-server, official release win-cuda-13.4-x64
- **Model (exact tag):** qwen3.8:27b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** 65,536 (completely on the GPU, 131,072 skipped by VRAM estimate: 25,609 MiB needed, 24,564 MiB total)
- **Cold load (s):** 5.32
- **TTFT cold (s):** 5.56
- **TTFT warm (s):** 2.53
- **TTFT with cache (s):** 0.24
- **Prompt processing tok/s:** 3,449
- **Generation tok/s:** 48.4
- **Generation tok/s at max context:** 40.6 (prompt processing 2,833)
- **Notes:** Without MTP, llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k. The first load took 9.4 s, then 5.3 s. In the prompt cache test, 7,691 tokens came from the cache. Uses 17.7 GB VRAM at 16k and 20.8 GB at 64k. llama-bench (`-p 512,8192 -n 256 -r 3`, without MTP): prompt processing 3,699 tok/s at 512 and 3,590 tok/s at 8,192 tokens, generation 50.5 tok/s, at a depth of 64,512 tokens prompt processing 2,247 and generation 41.1 tok/s.

<a id="ollama-qwen3-8-27b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** Ollama 0.40.13 (winget)
- **Model (exact tag):** qwen3.8:27b
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** 32,768 (100% on the GPU, 65,536 skipped by VRAM estimate: 23,528 MiB needed, at most 23,028 MiB allowed with 1.5 GB reserve)
- **Cold load (s):** 3.77
- **TTFT cold (s):** 4.13
- **TTFT warm (s):** 2.74
- **TTFT with cache (s):** 0.27
- **Prompt processing tok/s:** 3,219
- **Generation tok/s:** 96.5
- **Generation tok/s at max context:** 89.9 (prompt processing 2,999)
- **Notes:** Own `ollama serve` with one slot and f16 KV cache, base context 16k. Ollama uses MTP on Qwen3.8 (`--spec-type draft-mtp --spec-draft-n-max 4`) and loads the vision projector, so generation varies between runs (85.1 to 101.1 tok/s). The first load took 7.1 s, then 3.5 to 3.8 s. Uses 19.9 GB VRAM at 16k and 21.0 GB at 32k. Unlike on the RTX 3090, the 32k step ran without a hang or blue screen.

## Summary

The best setup on the RTX 4090 is Qwen3.8 27B with Ollama 0.40.13 (96.5 tok/s generation thanks to MTP, 3,219 tok/s prompt processing, up to 32k context), while llama.cpp without MTP generates only 48.4 tok/s but reaches 64k context. Unusual: both engines ran Qwen3.5 4B with the full 256k context without a GPU reset (the RTX 3090 with an older llama.cpp build reset at that step), and Ollama processes prompts on the 4B about 16 percent faster than llama.cpp, while llama.cpp generates about 9 percent faster. No run was aborted; only the first 256k attempt with llama.cpp failed because of a bug in the measuring script and was repeated successfully, and the context steps 128k (llama.cpp) and 64k (Ollama) on the 27B were skipped by the VRAM estimate.
