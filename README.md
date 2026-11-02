# LLM Benchmarks

Local language models on real hardware, measured with one prompt for every device. The input is 8,192 tokens of German prose and the answer is always 256 tokens long. **Prompt processing** shows how fast a document is read, **generation** shows how fast the answer is written.

## Results

| Device | Model | Engine | Prompt processing tok/s | Generation tok/s |
|:------|:-------|:-------|-----------------------:|-----------------:|
| **PC RTX 3090 24 GB** | Qwen3.5 4B Q4_K_M | llama.cpp 0.7.0-dev (build 11909) | 7,628 | 163.6 |
| | Qwen3.5 4B Q4_K_M | Ollama 0.40.3 | 6,626 | 149.8 |
| | Qwen3 14B Q4_K_M | Ollama 0.40.3 | 2,924 | 71.8 |
| | Qwen3 14B Q4_K_M | llama.cpp 0.7.0-dev (build 11909) | 2,782 | 71.4 |
| | Qwen3.8 27B Q4_K_M | llama.cpp 0.7.0-dev (build 11909) with MTP | 1,507 | 67.0 |
| | Qwen3.8 27B Q4_K_M | Ollama 0.40.3 | 1,351 | 66.4 |
| | Qwen3.8 27B Q4_K_M | llama.cpp 0.7.0-dev (build 11909) | 1,610 | 42.2 |
| | Flint 40B-A2.5B Q4_K_M | llama.cpp 0.7.0-dev (build 11909) | 3,183 | 121.7 |
| | Flint 40B-A2.5B UD-Q3_K_XL | llama.cpp 0.7.0-dev (build 11909) | 3,286 | 119.2 |
| | Flint 40B-A2.5B NVFP4 | Ollama 0.40.12 (custom build) | 2,512 | 117.0 |
| | LFM3 1.7B Q4_K_M | llama.cpp 0.8.0-dev (build 12001) | 18,882 | 307.4 |
| | LFM3 1.7B Q4_K_M | Ollama 0.40.13 | 20,863 | 285.3 |
| | Qwen3-Coder-30B-A3B Q4_K_M, abliterated | llama.cpp 0.8.0-dev (build 12001) | 4,408 | 162.4 |
| | Qwen3.8-Flash-Next IQ3_S | Strata 0.2.8 (engine 0.1.54) | 2,427 | 85.0 |
| **MacBook Pro M5 Max 128 GB** | Qwen3.5 4B Q4_K_M | Ollama 0.40.11 | 4,171 | 118.0 |
| **PC RTX 4060 Ti 8 GB** | Qwen3.5 4B Q4_K_M | llama.cpp b11889 (CUDA 13.4) | 5,420 | 72.2 |
| | Qwen3.5 4B Q4_K_M | Ollama 0.40.11 | 4,938 | 59.4 |
| | Qwen3 14B Q4_K_M | llama.cpp b11889 (CUDA 13.4) | does not fit | does not fit |
| | Qwen3 14B Q4_K_M | Ollama 0.40.11 | does not fit | does not fit |
| | Qwen3.8 27B Q4_K_M | llama.cpp b11889 (CUDA 13.4) | does not fit | does not fit |
| | Qwen3.8 27B Q4_K_M | Ollama 0.40.11 | does not fit | does not fit |
| **PC Radeon RX 560 4 GB** | Qwen3.5 4B Q4_K_M | llama.cpp 0.8.0-dev (build 12024, Vulkan) | 136 | 20.4 |
| | Qwen3.5 4B Q4_K_M | Ollama 0.40.13 | 135 | 18.3 |

"does not fit" means the model does not fit completely into GPU memory.

## Benchmark prompt

Paste the prompt into Claude Code or a similar agent on the test machine and answer its questions. Restart first and close as many other programs as you can.

```text
You are running a standardized benchmark for local language models on this machine. Goal: comparable measurements for the results table at https://github.com/maluramichael/llm-benchmarks. Work through the steps in order. Do not guess: check commands, flags and model tags first with --help, the installed version or the official docs. A short analysis with a time estimate comes first, then the user picks which tests run. Running only some of the tests is still a valuable contribution.

STEP 0: SHORT ANALYSIS, TIME ESTIMATE AND SELECTION (before anything else, read-only, 2 minutes at most)
The user should see up front which tests are possible on this machine and how long they take, and choose for themselves. Before the selection, do not install, download, quit or restart anything.
- If ~/llm-benchmark/<today's date>/selection.json already contains a selection (for example because the prompt was pasted again after the restart), show it with the estimated duration and only ask whether it still applies. If yes, continue directly with PREPARATION.
- Collect without asking: operating system, RAM, GPU and memory usable for models (commands as in STEP 1 and CRASH PROTECTION), free disk space, installed engines with version (ollama --version, llama-server --version, on Apple Silicon mlx-lm), models already present (ollama list, GGUF files in the Hugging Face cache and common model folders).
- Derive the possible tests from this: every combination of engine (STEP 2) and model class (STEP 3) that fits into usable memory. Mark classes that do not fit as "does not fit" right away, they cost no time.
- Check the results table in the repo for combinations that already exist for this device type. Mark missing combinations as "new" and offer them first. Repeat runs are still welcome.
- Estimate the duration per test as a range in minutes:
  - A full test processes about 75,000 prompt tokens (3 x 512 cold, 3 x 8,192 warm, 6 x 8,192 prompt cache) and generates 3,072 tokens (12 x 256), plus three times the load time. The context test reads each step once and the best step two more times: 8k to 128k are about 250,000 tokens in total, up to 256k about 500,000. From 64k on, assume half the prompt processing speed. On slow devices the context test takes the most time.
  - Take prompt processing and generation in tok/s from the results table: same device type, otherwise the most similar device plus 30 percent. Without a reference value, estimate cautiously and say so.
  - Reference values for an RTX 3090 with 24 GB: a full test (one engine, one model, with context test) took 6 to 10 minutes, llama-bench an extra 2 to 5 minutes, cutting the texts once per model 1 to 2 minutes.
  - List downloads separately with size in GB and duration. Measure the speed briefly (download a large Hugging Face file with curl for about 15 seconds, then abort), otherwise assume 100 Mbit/s and say so. Models already present cost no download time.
  - Add one-time costs: restart and cleanup about 10 minutes, each missing engine 5 to 10 minutes.
- Show the result as a short table: test (engine, model, class), new or already measured, download in GB, duration in minutes as a range. Below it, one line: "Total: about X to Y minutes".
- Then ask what should run (multiple choice, each option states duration and download):
  1. Models: the classes that fit, new combinations first and marked as recommended.
  2. Engines: Ollama, llama.cpp, on Apple Silicon also MLX.
  3. Scope per test: "Full" (all four measurements), "Without context test" (cold start, warm, prompt cache, usually saves half, max context is then "not measured") or "Warm only" (3 requests with 8,192 tokens, about 1 to 3 minutes per test). Also llama-bench yes or no.
  4. Time budget, if the selection takes longer than 60 minutes: run everything or cut to 30, 60 or 120 minutes. When cutting, keep new combinations and small models first.
- Have the user confirm the selection with the total duration in one sentence and save it as ~/llm-benchmark/<date>/selection.json (tests, scope, estimate per test). All following steps apply to this selection only: check or install only the chosen engines, load only the chosen models, take only the chosen measurements. For scope "Without context test" or "Warm only", state the scope in the notes and record missing values as "not measured".
- During the run, briefly report progress and remaining time after each test (elapsed time against the estimate). If a test takes more than twice as long as estimated, ask whether to continue. Record the actual duration per test in the JSON.

PREPARATION (after step 0, before step 1)
For the best results, as much memory as possible must be free. On Apple Silicon, CPU and GPU share the same memory. On Windows, browsers, Docker and connected monitors also use GPU memory.
- The machine must be freshly restarted. Read the time since the last boot (macOS: sysctl -n kern.boottime, Windows: (Get-CimInstance Win32_OperatingSystem).LastBootUpTime). If the boot was more than 30 minutes ago, ask the user to restart and paste the prompt again in a new session afterwards. The selection from step 0 stays in selection.json. On Windows choose "Restart", not "Shut down", otherwise Fast Startup keeps the old state. Check the CRASH PROTECTION items that need a restart themselves (TdrDelay) first, so one restart covers everything.
- Quit as many programs as possible, except the agent and its terminal. Above all Colima, Docker (on Windows then also wsl --shutdown), browsers, IDEs, chat programs and background apps in the menu bar or notification area (cloud sync, messengers, updaters). List running programs by memory use (macOS: ps -axo rss=,comm= | sort -rn | head -15, Windows: Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 Name,WorkingSet64) and ask the user to quit them. Only quit them yourself with consent, because unsaved work can be lost.
- If possible, unplug all external monitors, at the latest before step 4. Every display uses memory for frame buffers and window rendering. List connected monitors (macOS: system_profiler SPDisplaysDataType, Windows: Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorBasicDisplayParams) and point this out to the user. On a laptop, only the built-in display stays on. Operate a desktop PC without a monitor over SSH from another device. If it needs a monitor, connect it to the onboard graphics (see CRASH PROTECTION).
- Note the free memory after cleanup. If the user did not want to restart, kept a program running or left a monitor connected, record that in the notes of every test (for monitors: how many and on which graphics).

CRASH PROTECTION (applies to all steps)
If GPU memory fills up, the GPU can hang and cause a blue screen on Windows. Observed on an RTX 3090 during the max context test of Qwen3.8 27B with 32k context. Therefore:
- Windows with NVIDIA: Before starting, ask the user to set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control Panel under "Manage 3D settings". A full memory then gives a clean out-of-memory error instead of a hang. This cannot be set officially from the command line, so have the user confirm it.
- Windows: Read the registry key HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers. If TdrDelay and TdrDdiDelay are missing (default 2 seconds), recommend setting TdrDelay = 10 and TdrDdiDelay = 20 as DWORD and restarting. Only change this with consent, never set TdrLevel to 0.
- Ideally the monitor is connected to the CPU's onboard graphics, so the dedicated GPU computes without a desktop and the graphics watchdog rarely triggers. This requires an iGPU: Intel without "F" in the model name, AMD Ryzen up to series 5000 only with "G" (e.g. 5700G), from series 7000 all without "F". Check and point this out to the user, but do not insist.
- macOS: The GPU must not use all of the unified memory. The real limit is Metal's recommendedMaxWorkingSetSize. Read it with: echo 'import Metal; print(MTLCreateSystemDefaultDevice()!.recommendedMaxWorkingSetSize / 1048576)' > /tmp/mtl.swift && swift /tmp/mtl.swift (result in MiB), or from the recommendedMaxWorkingSetSize line in the llama-server log. Measured on a MacBook Pro M5 Max with 128 GB: 110,100 MiB, about 107.5 GB (84 percent). Model plus KV cache plus 1.5 GB reserve must stay below this, otherwise skip the test. Only change sysctl iogpu.wired_limit_mb with the user's consent (takes effect immediately, lasts until restart, 0 means default), leave at least 16 GB free for macOS and record the change in the notes.
- Check free GPU memory before every test (nvidia-smi --query-gpu=memory.used,memory.total --format=csv). If less than 1.5 GB stays free, skip the test and record "does not fit".
- Windows: After every test, check whether a GPU driver reset has happened since the test started (System log: source nvlddmkm event 153 or Display event 4101, or a new file under C:\Windows\LiveKernelReports\WATCHDOG). If so, end the whole run immediately, and do not continue with smaller context steps either. The blue screen can come only with the second hang. Ask the user to restart, then repeat the test with batch size 256 (Ollama num_batch, llama.cpp -ub) and record that in the notes.
- Save results as JSON immediately after each test, not at the end. A crash may only cost the running test.

STEP 1: CAPTURE THE SYSTEM
Reuse the values from step 0 and only add what is missing.
- Operating system and version, CPU, RAM.
- GPU and memory usable for models:
  - macOS: chip and unified memory (system_profiler SPHardwareDataType, sysctl hw.memsize). Only the Metal limit recommendedMaxWorkingSetSize is usable for models (see CRASH PROTECTION), enter it as "Memory for models".
  - NVIDIA: nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  - AMD: rocm-smi, or on Windows Get-CimInstance Win32_VideoController
- Free disk space. If a laptop is on battery, ask for the power supply to be connected.
- Build the device identifier following the pattern "<device type> <chip or GPU> <memory>", e.g. "MacBook Pro M5 Max 128 GB", "MacBook Air M1 16 GB", "PC RTX 3090 24 GB", and have the user confirm it.

STEP 2: ENGINES
Only the engines chosen in step 0.
- Ollama (all systems) and llama.cpp with llama-bench and llama-server (all systems). On Apple Silicon also MLX (mlx-lm), if the user agrees.
- Use existing installations. If something is missing, ask before installing and only use official channels (Homebrew, winget, official releases or installers, uv for Python). No sudo without asking.
- Record the version of each engine.

STEP 3: CHOOSE MODELS
Only the classes chosen in step 0.
Classes S to XL use the same model family so devices stay comparable: Qwen3.x, quantization Q4_K_M (MLX: 4 bit). Class F adds Flint as a second reference outside the Qwen family.
- Class S: smallest Qwen3.x model around 4B parameters.
- Class M: Qwen3.x model around 14B parameters.
- Class L: Qwen3.8 27B (the reference model).
- Class XL, only with at least 64 GB of usable memory: the largest Qwen3.x model that fits into usable memory with at least 20 percent reserve for the KV cache.
- Class F, only with at least 24 GB of usable memory: Flint by Recursive (40B parameters, Mixture of Experts with about 2.5B active parameters per token, up to 256k context, KV cache only about 28 KiB per token). Files per engine:
  - llama.cpp: Q4_K_M from ggml-org/Flint-GGUF (Flint-Q4_K_M.gguf, 23.4 GB). With less than 32 GB of usable memory, also UD-Q3_K_XL from unsloth/flint-GGUF (flint-UD-Q3_K_XL.gguf, 18.5 GB): on a 24 GB card, Q4_K_M fills the memory at 16k context already, while the 3-bit file reaches 64k at the same speed. Needs a llama.cpp build with Flint support (tested with build 11909, commit caa8191df), update older builds first.
  - Ollama: flint:40b, which is NVFP4, there is no Q4_K_M in the Ollama library. Needs Ollama 0.40.9 or newer. Check the quantization with ollama show and state it in the notes.
  - MLX: RecursiveSI/flint-int4 (INT4 weights from the vendor, 22.6 GB). If the installed mlx-lm version lacks Flint support, record MLX for class F as "not measured".
  Enter it in the card as class F, e.g. "Flint 40B-A2.5B Q4_K_M (F)". Cut the input texts with Flint's tokenizer, it differs from Qwen's.
Check exact tags in the Ollama library or the GGUF files on Hugging Face first. If a model does not exist, take the closest one from the same family and state that in the notes. Skip a class that does not fit completely into GPU memory and record "does not fit". Ask before any download over 20 GB and when disk space is tight, unless the download was already confirmed with its size in step 0.

STEP 4: MEASURE
The input is German prose from 17 Wikipedia articles (Rhein, Donau, Elbe, Hamburg, München and others), about 1.8 MB in total. Use German text so results stay comparable. Cut it to the required length with the tokenizer of the respective model (512 tokens, 8,192 tokens, and just below the context limit for the context test) and append the task "Fasse den obigen Text in wenigen Sätzen sachlich zusammen".
For each chosen combination of engine and model, within the chosen scope. Fixed settings: temperature 0, seed 42, 256 output tokens, each measurement 3 times, use the median. No other compute-heavy programs in parallel.
1. Cold start: fully unload the model (ollama stop or restart the server), then send a request with 512 input tokens. Measure: model load time in seconds and time to first token (TTFT cold).
2. Warm: the model stays loaded, new request with 8,192 input tokens (new text). Measure: TTFT warm, prompt processing in tokens per second, generation in tokens per second.
3. Prompt cache: the same 8,192-token prefix twice, each with a different short question at the end. Measure: TTFT of the second request (shows whether the prefix cache works).
4. Maximum context: increase the context size step by step (8k, 16k, 32k, 64k, 128k, up to the model's maximum). The largest value at which the model still runs completely on the GPU (Ollama: ollama ps shows 100% GPU) and a request with an almost full context completes. Measure prompt processing and generation in tokens per second at this value. Estimate per step beforehand whether model plus KV cache fit into free memory, and do not start steps that do not fit on paper. End the ladder immediately, never try the next step, as soon as a request fails, ollama ps shows less than 100% GPU, shared GPU memory (Windows: performance counter "GPU Process Memory", Shared Usage) rises by more than 500 MB over the previous step, less than 1.5 GB of GPU memory stays free, or on macOS swap (sysctl vm.swapusage) grows by more than 1 GB. Max context is then the last successful step.
Scope "Without context test": only 1 to 3. Scope "Warm only": only 2.
Sources for measurements: Ollama API (load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration), llama-bench (prompt and generation test, depth via the matching option per --help), llama-server with cache_prompt, mlx-lm per --help. Verify the token count of the input texts with the model's tokenizer.

STEP 5: RESULT
Save raw data as JSON under ~/llm-benchmark/<date>/. Missing values are written as "not measured".
a) Report as ~/llm-benchmark/<date>/results.md: table "System" (device, chip / GPU, memory for models, operating system, date), section "Tests" with one card per combination of engine and model. Card heading: "<engine> <version>, <model> <quantization> (<class>)". Fields in exactly this order: engine + version, model (exact tag), quantization, class, max context, cold load (s), TTFT cold (s), TTFT warm (s), TTFT with cache (s), prompt processing tok/s, generation tok/s, generation tok/s at max context, notes. At the end a section "Summary" in three sentences: best setup on this device, anything unusual, aborted runs. Do not put names, host names, IP addresses or user paths into the report.
b) Output one line per test for the results table, columns in exactly this order:
| Device | Model | Engine | Prompt processing tok/s | Generation tok/s |

STEP 6: CLEAN UP
Stop any servers and processes you started. Only delete downloaded models when asked.
```
