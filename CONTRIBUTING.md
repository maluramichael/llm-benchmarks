# Contributing

Every device makes the table more useful, and a single model on a single engine is already a valid run. Repeat runs on hardware that is already listed are welcome too.

There are two ways to send results:

1. **Pull request** with a run file (described below).
2. **Issue**, if you don't want to deal with git: open a [benchmark run issue](https://github.com/maluramichael/llm-benchmarks/issues/new?template=benchmark-run.yml) and paste your `results.md`. It gets turned into a run file for you.

## 1. Run the benchmark

1. Copy the prompt from [Benchmark prompt](README.md#benchmark-prompt) into Claude Code, Codex or a similar agent on the machine you want to test.
2. Answer its questions. It checks what fits into memory, estimates the duration and lets you pick the tests.
3. When it is done, the report is in `~/llm-benchmark/<date>/results.md`.

The prompt asks for a fresh restart and as few running programs as possible. If you skip that, it records it in the notes, which is fine. Results are still useful as long as the notes say what was different.

## 2. Add a run file

1. Fork the repository and create a branch.
2. Copy `results.md` to `runs/<date>-<device>.md`, for example `runs/2026-11-05-pc-rtx-5080-16gb.md`. The device part is the device identifier in lowercase with hyphens. A second run of the same device on the same day gets `-2` at the end.
3. Adjust the file to the format below (mostly the first line).
4. Open a pull request. The template contains a short checklist.

**Do not edit the results table, the chart or the run list in the README.** They are generated from `runs/*.md` after the merge. If you want to preview them, run `python3 scripts/build.py` (Python 3.10 or newer, no packages needed) and then discard the generated changes with `git checkout README.md charts/`, otherwise they conflict with other pull requests.

## Run file format

The build script reads the run files, so a few lines have to follow a fixed pattern. Everything else is free text.

```markdown
# 2026-11-05: PC RTX 5080 16 GB

One run of the standard benchmark. Results table and prompt are in the [README](../README.md).

## System

| | |
|:--|:--|
| **Device** | PC RTX 5080 16 GB |
| **Chip / GPU** | ... |
| **Memory for models** | ... |
| **Operating system** | ... |
| **Date** | 2026-11-05 |

## Tests

<a id="ollama-qwen3-5-4b-q4-k-m"></a>
### Ollama 0.40.13, Qwen3.5 4B Q4_K_M (S)

- **Engine + version:** Ollama 0.40.13
- **Model (exact tag):** qwen3.5:4b
- **Quantization:** Q4_K_M
- **Class:** S
- **Max context:** 262,144
- **Cold load (s):** 1.78
- **TTFT cold (s):** 1.91
- **TTFT warm (s):** 0.51
- **TTFT with cache (s):** 0.11
- **Prompt processing tok/s:** 18,310
- **Generation tok/s:** 185.8
- **Generation tok/s at max context:** 69.7
- **Notes:** ...

## Summary

Three sentences: best setup, anything unusual, aborted runs.
```

The rules behind it:

- **First line:** `# YYYY-MM-DD: <device>`. The date must match the start of the file name. The device follows the pattern `<device type> <chip or GPU> <memory>`, for example `MacBook Air M1 16 GB` or `PC RTX 3090 24 GB`. Use exactly the same name as existing runs of that device, otherwise it shows up as a separate device.
- **Test cards** live under `## Tests`. Each heading reads `### <engine> <version>, <model> <quantization> (<class>)`. The class is one of `S`, `M`, `L`, `XL`, `F` or `Extra` (see step 3 of the prompt).
- **Numbers** are read from `Prompt processing tok/s` and `Generation tok/s`. Thousands separators are fine. If a model does not fit, write `does not fit` under `Max context`. Missing values are `not measured`.
- **The anchor line** (`<a id="..."></a>`) is optional. Without it, the link in the results table points to the heading itself.
- **Variants** of an engine are written as `<engine> <version> with <variant>`, for example `llama.cpp 0.7.0-dev (build 11909) with MTP`. They appear in the table but not in the chart, so the chart always compares the plain engines.

If something does not match, `python3 scripts/build.py` stops with a message that names the file and the line.

## License

The repository is licensed under [CC BY 4.0](LICENSE). By opening a pull request or a benchmark run issue you agree that your contribution is published under the same license.

## Privacy

Do not put names, host names, IP addresses, user paths or serial numbers into the run file. The prompt avoids this already, but check before you push.

## Other changes

Changes to the benchmark prompt affect whether new runs stay comparable with the old ones. Please open an issue first and describe what you want to change and why. Suggestions for models, engines or tests are welcome as a [suggestion issue](https://github.com/maluramichael/llm-benchmarks/issues/new?template=suggestion.yml).
