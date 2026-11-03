# Contributing

Every device makes the charts more useful, and a single model on a single engine is already a valid run. Repeat runs on hardware that is already listed are welcome too.

There are two ways to send results:

1. **Pull request** with a run folder (described below).
2. **Issue**, if you don't want to deal with git: open a [benchmark run issue](https://github.com/maluramichael/llm-benchmarks/issues/new?template=benchmark-run.yml) and paste the `README.md` of your run. It gets turned into a run folder for you.

## 1. Run the benchmark

Either give the [agent prompt](README.md#agent-prompt) to Claude Code or a similar agent, or do it yourself:

```text
./bench.sh doctor
./bench.sh plan
./bench.sh run
./bench.sh report
```

On Windows use `.\bench.ps1`. Lists are comma separated (`--models S,M`), and PowerShell users can type them without quotes. Restart the machine first and close other programs. `doctor` tells you what it finds.

- A run continues where it stopped. If it was interrupted, start the same command again.
- A new run on the same device continues the newest folder of that device and skips tests that are finished. To measure again, use `--force-test <pattern>` or `--force`. A new folder with `-2` at the end appears only when the old folder cannot be continued (unreadable or an old hand-made run).
- `--force-test <pattern>` measures matching tests again (`--force` measures everything again), for example `--force-test "llamacpp/S/speed/*"`.
- Run `bench --help` and `bench run --help` for all options.

The results are in `runs/<date>-<device>/`. It holds `results.json` (the data), `README.md` (the report, written by `bench run` and `bench report`) and `raw/` (the raw answers the report links to).

## 2. Send the run folder

1. Fork the repository and create a branch.
2. Add your whole run folder from `runs/`, including `raw/`. The report links to it. Do not rename the folder.
3. Open a pull request. The template has a short checklist.

**Do not edit the charts or the run list in the README.** They are generated from `runs/*/results.json` after the merge. To preview them, run `python3 scripts/build.py` and then discard the generated changes with `git checkout README.md charts/`, otherwise they conflict with other pull requests.

## Privacy

`bench report` refuses to write a report when it finds private data, for example user names, host names, IP addresses, network paths or user paths. It does not look for serial numbers, so check for those yourself. If it does, fix the file it names and run it again.

Engine logs stay in `.bench/logs` on your machine. Do not add them to the pull request. Do not add anything from `.bench/` at all.

## Platforms

Windows with NVIDIA is tested. Linux with AMD (Vulkan) is being tested. macOS with Metal and AMD with ROCm are untested: the code has a path for them, but no run exists yet. Runs from these machines are very welcome, and if something breaks, open an issue with the output of `bench doctor`.

## Old runs

Runs marked "old prompt" in the README were measured by hand before `bench` existed. Their folders only hold a `README.md` and a small `results.json` with `"legacy": true`. They are not in the charts. Do not add new runs in this form.

## License

The repository is licensed under [CC BY 4.0](LICENSE). By opening a pull request or a benchmark run issue you agree that your contribution is published under the same license.

## Other changes

Changes to the tasks, the corpus or the fixed settings affect whether new runs stay comparable with old ones. Please open an issue first and describe what you want to change and why. Suggestions for models, engines or tests are welcome as a [suggestion issue](https://github.com/maluramichael/llm-benchmarks/issues/new?template=suggestion.yml).
