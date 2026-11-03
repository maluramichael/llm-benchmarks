# 2026-10-26: PC RTX 3090 24 GB

One run of the standard benchmark. Results table and prompt are in the [README](../../README.md).

## System

| | |
|:--|:--|
| **Device** | PC RTX 3090 24 GB |
| **Chip / GPU** | NVIDIA GeForce RTX 3090 (driver 610.88), AMD Ryzen 9 5950X, 128 GB RAM |
| **Memory for models** | 24 GB VRAM |
| **Operating system** | Windows 11 IoT Enterprise LTSC (build 26100) |
| **Date** | 2026-10-26 |

## Tests

<a id="ollama-qwen3-5-4b-q4-k-m"></a>
### Ollama 0.40.3, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.3
- **Model (exact tag):** qwen3.5:4b
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 262,144 (model maximum, completely on the GPU)
- **Cold load (s):** 2.67
- **TTFT cold (s):** 2.86
- **TTFT warm (s):** 1.30
- **TTFT with cache (s):** 0.26
- **Prompt processing tok/s:** 6,626
- **Generation tok/s:** 149.8
- **Generation tok/s at max context:** 60.5 (prompt processing 2,527)
- **Notes:** Standard prompt, median of 3 runs, raw text completion, base context 16k.

<a id="ollama-qwen3-14b-q4-k-m"></a>
### Ollama 0.40.3, Qwen3 14B Q4_K_M (M)

- **Engine + version:** Ollama 0.40.3
- **Model (exact tag):** qwen3:14b
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** 40,960 (model maximum, completely on the GPU)
- **Cold load (s):** 2.89
- **TTFT cold (s):** 3.15
- **TTFT warm (s):** 2.86
- **TTFT with cache (s):** 0.07
- **Prompt processing tok/s:** 2,924
- **Generation tok/s:** 71.8
- **Generation tok/s at max context:** 49.7 (prompt processing 1,716)
- **Notes:** Class M uses Qwen3 instead of Qwen3.x. A first pass with 4 parallel Ollama slots was only 72.6% on the GPU at 32k context (2.8 GB in shared memory) and was discarded. The values come from the second pass with one slot.

<a id="ollama-qwen3-8-27b-q4-k-m"></a>
### Ollama 0.40.3, Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** Ollama 0.40.3
- **Model (exact tag):** qwen3.8:27b
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** 16,384 (completely on the GPU, larger steps blocked after two blue screens at 32,768)
- **Cold load (s):** 4.90
- **TTFT cold (s):** 5.39
- **TTFT warm (s):** 6.18
- **TTFT with cache (s):** 0.59
- **Prompt processing tok/s:** 1,351
- **Generation tok/s:** 66.4
- **Generation tok/s at max context:** 62.8 (prompt processing 1,296)
- **Notes:** Repeat run with crash protection: own `ollama serve` with one slot and f16 KV cache, free VRAM check before each context step, GPU reset check after each test. At 16,384 context an estimated 22.1 of 22.9 GB were in use, so 32,768 would not have fit. The first cold start took 17.3 s (model not yet in the file cache), then 4.9 s. In the prompt cache test, 7,699 of 8,210 tokens came from the cache. Ollama uses MTP on Qwen3.8 (speculative decoding, 4 draft tokens). The first run on the same day aborted at 32k with a GPU hang and blue screen 0x7E.

<a id="llama-cpp-qwen3-5-4b-q4-k-m"></a>
### llama.cpp 0.7.0-dev (build 11909), Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** llama.cpp 0.7.0-dev, build 11909 (commit caa8191df), llama-server, self-built with CUDA
- **Model (exact tag):** qwen3.5:4b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 131,072 (crash at 262,144, see notes)
- **Cold load (s):** 3.35
- **TTFT cold (s):** 3.46
- **TTFT warm (s):** 1.11
- **TTFT with cache (s):** 0.13
- **Prompt processing tok/s:** 7,628
- **Generation tok/s:** 163.6
- **Generation tok/s at max context:** 92.4 (prompt processing 4,083)
- **Notes:** llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k, median of 3 runs. In the prompt cache test, 7,695 tokens came from the cache and only 515 were processed again. At 262,144 context, llama.cpp aborted after about 133,000 processed tokens with a CUDA error (illegal memory access) and Windows reported a GPU reset (nvlddmkm 153). The script ended the run immediately, there was no blue screen. Only an estimated 12.9 of 22.9 GB were in use, so a memory overflow is unlikely. llama-bench (`-p 512,8192 -n 256`, 3 repetitions): prompt processing 8,019 tok/s at 512 and 7,806 tok/s at 8,192 tokens, generation 177.8 tok/s, at a depth of 130,048 tokens prompt processing 2,535 and generation 95.1 tok/s.

<a id="llama-cpp-qwen3-14b-q4-k-m"></a>
### llama.cpp 0.7.0-dev (build 11909), Qwen3 14B Q4_K_M (M)

- **Engine + version:** llama.cpp 0.7.0-dev, build 11909 (commit caa8191df), llama-server, self-built with CUDA
- **Model (exact tag):** qwen3:14b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** M
- **Max context:** 40,960 (model maximum, completely on the GPU)
- **Cold load (s):** 10.34
- **TTFT cold (s):** 10.57
- **TTFT warm (s):** 2.99
- **TTFT with cache (s):** 0.05
- **Prompt processing tok/s:** 2,782
- **Generation tok/s:** 71.4
- **Generation tok/s at max context:** 49.7 (prompt processing 1,629)
- **Notes:** Class M uses Qwen3 instead of Qwen3.x. llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k, median of 3 runs. In the prompt cache test only 16 tokens were processed again, the rest came from the cache. Loading takes much longer than with Ollama (10.3 s instead of 2.9 s). llama-bench (`-p 512,8192 -n 256`, 3 repetitions): prompt processing 3,296 tok/s at 512 and 2,782 tok/s at 8,192 tokens, generation 81.4 tok/s, at a depth of 39,936 tokens prompt processing 1,073 and generation 50.6 tok/s.

<a id="llama-cpp-qwen3-8-27b-q4-k-m"></a>
### llama.cpp 0.7.0-dev (build 11909), Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** llama.cpp 0.7.0-dev, build 11909 (commit caa8191df), llama-server, self-built with CUDA
- **Model (exact tag):** qwen3.8:27b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** 16,384 (completely on the GPU, larger steps blocked as with Ollama)
- **Cold load (s):** 18.51
- **TTFT cold (s):** 18.91
- **TTFT warm (s):** 5.17
- **TTFT with cache (s):** 0.47
- **Prompt processing tok/s:** 1,610
- **Generation tok/s:** 42.2
- **Generation tok/s at max context:** 41.4 (prompt processing 1,534)
- **Notes:** Without MTP, llama-server with `-ngl 99 -fa on --fit off -np 1`, base context 16k, median of 3 runs. Processes prompts faster than Ollama (1,610 instead of 1,351 tok/s), but generates much slower without MTP (42.2 instead of 66.4 tok/s), see the next card for MTP. In the prompt cache test, 7,695 tokens came from the cache. llama-bench (`-p 512,8192 -n 256`, 3 repetitions, without MTP): prompt processing 1,681 tok/s at 512 and 1,632 tok/s at 8,192 tokens, generation 44.2 tok/s, at a depth of 15,360 tokens prompt processing 1,427 and generation 42.1 tok/s.

<a id="llama-cpp-mtp-qwen3-8-27b-q4-k-m"></a>
### llama.cpp 0.7.0-dev (build 11909) with MTP, Qwen3.8 27B Q4_K_M (L)

- **Engine + version:** llama.cpp 0.7.0-dev, build 11909 (commit caa8191df), llama-server, self-built with CUDA, MTP on
- **Model (exact tag):** qwen3.8:27b (same GGUF file as Ollama)
- **Quantization:** Q4_K_M
- **Class:** L
- **Max context:** not measured
- **Cold load (s):** 18.47
- **TTFT cold (s):** 18.93
- **TTFT warm (s):** 5.52
- **TTFT with cache (s):** 0.51
- **Prompt processing tok/s:** 1,507
- **Generation tok/s:** 67.0
- **Generation tok/s at max context:** not measured
- **Notes:** Extra measurement with MTP (`--spec-type draft-mtp --spec-draft-n-max 4`, same as Ollama), otherwise like the card without MTP, no context test. MTP raises generation from 42.2 to 67.0 tok/s, about 60 percent, and costs some prompt processing speed (1,507 instead of 1,610 tok/s).

## Summary

The best setup on the RTX 3090 is Qwen3.8 27B with Ollama 0.40.3 (66.4 tok/s generation, 1,351 tok/s prompt processing, up to 16k context) or, about equal, llama.cpp with MTP (67.0 tok/s, 1,507 tok/s prompt processing), though Ollama loads in 4.9 instead of 18.5 seconds. MTP raises generation on the 27B by about 60 percent, and llama.cpp without MTP processes prompts faster but generates much slower. Aborted runs: the context test above 16k on the 27B (two blue screens at 32k, blocked since then) and step 262,144 with llama.cpp on Qwen3.5 4B (CUDA error with GPU reset, run ended immediately).
