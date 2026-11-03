"""Command line interface: bench system|doctor|plan|setup|run|report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bench import plan as planning
from bench import preflight, report, runner, safety
from bench.config import load_config
from bench.privacy import PrivacyError, local_secrets, scrub
from bench.store import ResultStore, StoreError
from bench.system import collect_system, suggest_device_name


def _system_command() -> int:
    info = collect_system()
    output = {"suggested_device_name": suggest_device_name(info), "system": info}
    print(json.dumps(scrub(output, local_secrets()), indent=2, ensure_ascii=False))
    return 0


def _device(cfg, system: dict | None = None) -> str:
    name = cfg.device or suggest_device_name(system if system is not None else collect_system())
    return scrub(name, local_secrets())


def _doctor_command(cfg) -> int:
    """Show the environment checks and the pre-flight findings. Changes nothing."""
    for row in safety.doctor(cfg):
        print(f"[{row['status']:4}] {row['check']}: {row['text']}")
        if row.get("detail"):
            print(f"       {row['detail']}")
    findings = preflight.check(cfg, cfg.gpu or 0)
    for finding in findings:
        print(f"[warn] {finding['kind']}: {finding['text']}")
    if not findings:
        print("[ok  ] nothing else uses the GPU, no foreign Ollama server on port 11434")
    return 0


def _plan_command(cfg) -> int:
    system = collect_system()
    store = None
    folder = runner.find_run_dir(cfg.output, _device(cfg, system), runner.Env().today())
    if (folder / "results.json").is_file():
        store = ResultStore(folder / "results.json")
    print(planning.format_plan(planning.make_plan(cfg, system, store=store)))
    return 0


def _report_command(cfg) -> int:
    """Render README.md of an existing run. `--output` may be the run folder itself or the
    folder that holds the runs (then the run of this device is used)."""
    output = Path(cfg.output)
    if (output / "results.json").is_file():
        folder = output
    else:
        folder = runner.find_run_dir(output, _device(cfg), runner.Env().today())
    if not (folder / "results.json").is_file():
        print(f"error: no results.json found for this device below {output}; "
              "run `bench run` first or point --output at the run folder")
        return 1
    try:
        report.write_report(folder)
    except PrivacyError as err:
        print("error: the report was not written, personal data was found:")
        for leak in err.leaks:
            print(f"  {leak}")
        print("Fix results.json or the files in raw/ by hand, then run `bench report` again.")
        return 1
    except (OSError, ValueError) as err:
        print(f"error: {err}")
        return 1
    print(f"Report: {folder / 'README.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    command, cfg = load_config(sys.argv[1:] if argv is None else argv)
    try:
        if command == "system":
            return _system_command()
        if command == "doctor":
            return _doctor_command(cfg)
        if command == "plan":
            return _plan_command(cfg)
        if command == "run":
            return runner.run(cfg)
        if command == "report":
            return _report_command(cfg)
    except StoreError as err:
        print(f"error: {scrub(str(err), local_secrets())}")
        return 1
    print(f"bench {command}: not implemented yet")
    return 0
