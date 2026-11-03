"""Runner: runs a whole benchmark (`bench run`).

Flow: collect the system, pre-flight and doctor checks, pick the run folder
`<output>/<first-start-date>-<device-slug>/` (an earlier folder of the same device is
continued, so a resume on the next day lands in the same place), open results.json,
recover tests a crash left `running`, then for every engine (try/finally stop) and every
model the suites in the fixed order memory -> speed -> smoke -> concurrency (later suites
read the memory ladder). At the end README.md is rendered from results.json.

Exit codes: 0 all done, 1 problems (a model or engine could not be used, a suite failed,
tests ended with `error`, privacy check failed), 3 aborted after a GPU driver reset,
130 interrupted with Ctrl+C. 2 is left to the argument parser. Every one of them has
saved what was measured; the same command continues where it stopped.

Only processes started by bench are stopped, nothing outside the output folder and the
engine caches is written, no system setting is changed.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from bench import VERSION, awake, corpus, models, plan, preflight, report, safety
from bench.config import Config
from bench.privacy import PrivacyError, local_secrets, scrub
from bench.store import FINAL, ResultStore, StoreError
from bench.suites import RunAborted, RunContext, clean_error
from bench.system import collect_system, suggest_device_name

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_ABORTED = 3
EXIT_INTERRUPTED = 130

SUITE_ORDER = ("memory", "speed", "smoke", "concurrency")
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_CACHE = REPO_ROOT / ".bench" / "cache" / "corpus"
LOG_DIR = REPO_ROOT / ".bench" / "logs"
MIB = 1024 * 1024
RUN_DIR_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
NUMBER_SUFFIX = re.compile(r"-(\d{1,2})$")  # -2, -3 ...: a folder name that was taken


# --- injectable pieces ---------------------------------------------------------------------

def default_engine(name: str, cfg: Config):
    if name == "ollama":
        from bench.engines.ollama import OllamaEngine
        return OllamaEngine(cfg, log_dir=LOG_DIR)
    from bench.engines.llamacpp import LlamaCppEngine
    return LlamaCppEngine(cfg, log_dir=LOG_DIR)


def default_corpus(cfg: Config) -> str:
    return corpus.load_corpus(CORPUS_CACHE, offline=cfg.no_download, progress=print)


def default_suites() -> dict[str, Callable]:
    from bench.suites.concurrency import run_concurrency
    from bench.suites.memory import run_memory
    from bench.suites.smoke import run_smoke
    from bench.suites.speed import run_speed
    return {"memory": run_memory, "speed": run_speed, "smoke": run_smoke,
            "concurrency": run_concurrency}


@dataclass
class Env:
    """Everything the runner takes from the outside. The defaults are the real system;
    tests replace single entries."""
    system: Callable[[], dict] = collect_system
    doctor: Callable[[Config], list[dict]] = safety.doctor
    preflight: Callable[[Config, int], list[dict]] = preflight.check
    make_engine: Callable = default_engine
    resolve: Callable = models.resolve
    guard: Callable = safety.Guard
    corpus: Callable[[Config], str] = default_corpus
    today: Callable[[], date] = lambda: datetime.now().date()
    keep_awake: Callable = awake.keep_awake
    say: Callable[[str], None] = lambda message: print(message, flush=True)
    suites: dict = field(default_factory=default_suites)


# --- run folder ----------------------------------------------------------------------------

def slugify(device: str) -> str:
    """Lower case, "12 GB" becomes "12gb", everything else that is not a letter or a digit
    becomes one dash."""
    text = re.sub(r"(\d)\s+(gb)\b", r"\1\2", device.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "device"


def run_dir_name(day: date, device: str) -> str:
    return f"{day.isoformat()}-{slugify(device)}"


def _continuable(folder: Path) -> bool:
    """A folder with a readable, current results.json of a real (not legacy) run. Anything
    else is foreign: it is left alone and never continued."""
    try:
        data = ResultStore(folder / "results.json").data
    except StoreError:
        return False
    return (folder / "results.json").is_file() and data.get("legacy") is not True


def _free_name(folder: Path) -> Path:
    candidate, number = folder, 2
    while candidate.exists():
        candidate = folder.with_name(f"{folder.name}-{number}")
        number += 1
    return candidate


def find_run_dir(output: Path, device: str, today: date) -> Path:
    """The folder of the newest earlier run of this device (any date). Legacy runs and
    folders without a readable results.json are never continued. Otherwise a new folder
    named after `today` (with -2, -3 ... when that name is taken)."""
    output = Path(output)
    slug = slugify(device)
    found = []
    if output.is_dir():
        for folder in output.iterdir():
            match = RUN_DIR_NAME.match(folder.name)
            if not (folder.is_dir() and match):
                continue
            rest, number = match.group(2), 1
            if rest != slug:
                suffix = NUMBER_SUFFIX.search(rest)
                if not suffix or rest[:suffix.start()] != slug:
                    continue
                number = int(suffix.group(1))
            if _continuable(folder):
                found.append((match.group(1), number, folder))
    if found:
        return sorted(found, key=lambda f: f[:2])[-1][2]
    return _free_name(output / run_dir_name(today, device))


# --- helpers -------------------------------------------------------------------------------

def _clean(value):
    return scrub(value, local_secrets())


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpu_of(system: dict, index: int) -> dict:
    gpus = [g for g in system.get("gpus") or [] if isinstance(g, dict)]
    for gpu in gpus:
        if gpu.get("index") == index:
            return gpu
    return gpus[index] if 0 <= index < len(gpus) else {}


def _stored_doctor_rows(rows: list[dict]) -> list[dict]:
    """check, status and text only. `detail` holds process names. The text of the process
    warning names big processes too, so it is replaced."""
    stored = []
    for row in rows:
        text = str(row.get("text", ""))
        if row.get("check") == "processes" and row.get("status") == "warn":
            text = "large processes were running at the start (names not stored)"
        stored.append({"check": str(row.get("check", "")), "status": str(row.get("status", "")),
                       "text": _clean(text)})
    return stored


@dataclass
class _State:
    cfg: Config
    env: Env
    store: ResultStore
    guard: object
    system: dict
    corpus_text: str
    raw_dir: Path
    notes: list[str]
    models_info: dict
    engines_info: dict
    problems: list[str] = field(default_factory=list)

    def say(self, message: str) -> None:
        self.env.say(message)

    def problem(self, message: str) -> None:
        self.problems.append(message)
        self.say(f"problem: {message}")

    def fail(self, message: str) -> None:
        """A problem that also goes into the notes, so results.json says why."""
        self.problem(message)
        self.note(message)

    def note(self, text: str) -> None:
        text = _clean(text)
        if text not in self.notes:
            self.notes.append(text)
            try:
                self.store.set_section("notes", self.notes)
            except OSError as err:
                self.say(f"warning: could not save a note: {_clean(str(err))}")


def _dict_of(value) -> dict:
    return value if isinstance(value, dict) else {}


def _first_gpu(system: dict) -> dict:
    gpus = [g for g in _dict_of(system).get("gpus") or [] if isinstance(g, dict)]
    return gpus[0] if gpus else {}


def _warn_changes(state: _State, run_info: dict, first: dict, now: dict) -> None:
    """Results from a changed setup in one run folder are not comparable: say so."""
    old, new = _first_gpu(first), _first_gpu(now)
    changes = []
    for label, was, is_now in (("GPU", old.get("name"), new.get("name")),
                               ("GPU driver", old.get("driver"), new.get("driver")),
                               ("bench version", run_info.get("bench_version"), VERSION)):
        if was and is_now and was != is_now:
            changes.append(f"the {label} changed since the run started (was {was}, "
                           f"now {is_now})")
    for message in changes:
        state.say(f"warning: {message}")
        state.note(message)


def _announce(store: ResultStore, say: Callable[[str], None]) -> None:
    """One line when a step starts and one when it ends ("<key> ... ok 12 s"), so a log
    shows where the run is. Wraps the two methods of this store object only."""
    started: dict[str, float] = {}
    mark_running, save = store.mark_running, store.save

    def marked(key, meta):
        started[key] = time.monotonic()
        say(f"{key} ...")
        return mark_running(key, meta)

    def saved(key, status, values, meta=None):
        result = save(key, status, values, meta)
        began = started.pop(key, None)
        took = f" {time.monotonic() - began:.0f} s" if began is not None else ""
        say(f"{key} ... {status}{took}")
        return result

    store.mark_running, store.save = marked, saved


def _existing_notes(store: ResultStore) -> list[str]:
    notes = store.data.get("notes")
    if isinstance(notes, str):
        return [notes]
    return [n for n in notes or [] if isinstance(n, str)]


# --- engines and models --------------------------------------------------------------------

def _record_engine(state: _State, engine) -> None:
    info = {}
    for name, call in (("version", engine.version), ("backend", engine.backend)):
        try:
            info[name] = call()
        except Exception:
            info[name] = None
    try:
        info.update(dict(getattr(engine, "meta", None) or {}))
    except Exception:
        pass
    info["thinking"] = bool(state.cfg.thinking)
    before = _dict_of(state.engines_info.get(engine.name)).get("version")
    if before and info.get("version") and before != info["version"]:
        message = (f"the engine version of {engine.name} changed since the run started "
                   f"(was {before}, now {info['version']})")
        state.say(f"warning: {message}")
        state.note(message)
    state.engines_info[engine.name] = _clean(info)
    state.store.set_section("engines", state.engines_info)


def _resolve_models(state: _State, engine_name: str) -> list:
    cfg, store = state.cfg, state.store
    resolved = []
    gpu = _gpu_of(state.system, cfg.gpu or 0)
    total = int(gpu.get("vram_total_mb") or 0)
    for model_id in cfg.models:
        label = f"{engine_name}/{model_id}"
        try:
            ref = state.env.resolve(model_id, engine_name, cfg, say=state.say)
        except models.NeedsConsent as err:
            state.fail(f"{label}: {err}")
            continue
        except models.ModelError as err:
            state.fail(f"{label}: {_clean(str(err))}")
            continue
        except RunAborted:
            raise
        except Exception as err:  # e.g. the NAS share is gone
            state.fail(f"{label}: cannot resolve the model: {clean_error(err)}")
            continue
        weights = (ref.size_bytes or 0) / MIB
        reserve = state.guard.reserve_mb()
        if total and weights and plan.vram_is_reliable(gpu) and weights + reserve > total:
            reason = (f"the weights alone ({weights / 1024:.1f} GB) plus the {reserve / 1024:.1f} "
                      f"GB safety reserve exceed the {total / 1024:.1f} GB of the card")
            key = f"{engine_name}/{ref.id}/memory/fit"
            entry = store.get(key)
            if entry is None or entry["status"] != "does_not_fit":
                store.save(key, "does_not_fit", {"reason": reason})
            state.say(f"{label}: does not fit, not run ({reason})")
            continue
        resolved.append(ref)
    return resolved


def _run_model(state: _State, engine, ref) -> bool:
    """Run the selected suites for one model. False: the engine should not be used further."""
    cfg = state.cfg
    label = f"{engine.name}/{ref.id}"
    state.models_info[label] = _clean({
        "engine": engine.name, "id": ref.id, "cls": ref.cls, "quant": ref.quant,
        "size_bytes": ref.size_bytes, "digest": ref.digest})
    state.store.set_section("models", state.models_info)
    ctx = RunContext(cfg, engine, ref, state.store, state.guard, state.corpus_text,
                     state.raw_dir)
    for name in SUITE_ORDER:
        if name not in cfg.suites:
            continue
        state.say(f"{label}: {name} ...")
        try:
            state.env.suites[name](ctx)
        except (RunAborted, KeyboardInterrupt):
            raise
        except Exception as err:
            _retryable(state, f"the {name} suite stopped with an unexpected error")
            state.problem(f"{label}: the {name} suite failed: {clean_error(err)}")
            state.note(f"{label}: the {name} suite stopped with an unexpected error "
                       f"({clean_error(err)})")
            return False
        state.say(f"{label}: {name} done")
    return True


def _run_engine(state: _State, engine_name: str) -> None:
    refs = _resolve_models(state, engine_name)
    if not refs:
        return
    try:
        engine = state.env.make_engine(engine_name, state.cfg)
    except Exception as err:
        state.fail(f"{engine_name}: cannot create the engine: {clean_error(err)}")
        return
    try:
        try:
            engine.start()
        except Exception as err:
            state.fail(f"{engine_name}: cannot start: {clean_error(err)}")
            return
        for ref in refs:
            if not _run_model(state, engine, ref):
                break
    finally:
        try:
            _record_engine(state, engine)
        except Exception:
            pass
        try:
            engine.stop()
        except Exception as err:
            state.say(f"warning: stopping {engine_name} failed: {clean_error(err)}")


def _retryable(state: "_State", reason: str) -> None:
    """Turn tests left `running` into `error`. Neither Ctrl+C nor a Python exception is a
    crash: recover_crashed would call them `aborted`, which for a ladder rung means
    "does not fit, never retried"."""
    for key, entry in list(state.store.data["tests"].items()):
        if entry.get("status") == "running":
            try:
                state.store.save(key, "error", {"reason": reason})
            except Exception as err:  # e.g. the disk is full: report it, do not crash
                state.problem(f"could not mark {key} as retryable: {clean_error(err)}")
                return


def _mark_interrupted(state: "_State") -> None:
    _retryable(state, "interrupted by the user (Ctrl+C); it is measured again by the next run")


# --- run -----------------------------------------------------------------------------------

def run(cfg: Config, env: Env | None = None) -> int:
    """Run the benchmark. Returns the exit code (see the module docstring)."""
    env = env or Env()
    try:
        return _run(cfg, env)
    except StoreError as err:
        env.say(f"error: {_clean(str(err))}")
        return EXIT_PROBLEMS
    except KeyboardInterrupt:
        env.say("Interrupted. Run the same command again to continue.")
        return EXIT_INTERRUPTED


def _run(cfg: Config, env: Env) -> int:
    say = env.say
    system = env.system()
    device = _clean(cfg.device or suggest_device_name(system))
    gpu_index = cfg.gpu or 0
    output = Path(cfg.output)
    run_dir = find_run_dir(output, device, env.today())

    if cfg.dry_run:
        existing = ResultStore(run_dir / "results.json") if (
            run_dir / "results.json").is_file() else None
        say(plan.format_plan(plan.make_plan(cfg, system, store=existing, resolve=env.resolve)))
        say("\nDry run: nothing was measured, copied or downloaded.")
        return EXIT_OK

    findings = env.preflight(cfg, gpu_index)
    for finding in findings:
        say(f"warning: {finding['text']}")
        if finding.get("detail"):
            say(f"  {finding['detail']}")
    if findings and cfg.abort_if_busy:
        say("Ending before the run because --abort-if-busy is set. Close what is using the "
            "GPU and run the same command again.")
        return EXIT_PROBLEMS

    try:
        corpus_text = env.corpus(cfg)
    except Exception as err:
        say(f"error: the text corpus is not available: {_clean(str(err))}")
        return EXIT_PROBLEMS

    store = ResultStore(run_dir / "results.json")
    _announce(store, say)
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    say(f"Run folder: {run_dir}")

    gpu = _gpu_of(system, gpu_index)
    guard = env.guard(cfg, gpu_index, int(gpu.get("vram_total_mb") or 0))
    state = _State(cfg, env, store, guard, system, corpus_text, raw_dir, _existing_notes(store),
                   dict(store.data.get("models") or {}), dict(store.data.get("engines") or {}))

    started = dict(store.data.get("run") or {})
    if store.data["tests"]:
        say(f"continuing run {run_dir.name} started {started.get('started_at', 'earlier')} "
            f"with {len(store.data['tests'])} results")
    started.setdefault("device", device)
    started.setdefault("started_at", _utc_now())
    started.setdefault("bench_version", VERSION)
    clean_system = _clean(system)
    first = store.data.get("system_first")
    if first is None:  # runs from before system_first existed: the stored system was the first
        first = store.data.get("system") or clean_system
        store.set_section("system_first", first)
    _warn_changes(state, started, first, clean_system)
    store.set_section("run", started)
    store.set_section("system", clean_system)
    for key in store.recover_crashed():
        state.note(f"{key} was still running when an earlier run ended (crash or power loss); "
                   "it was marked aborted")
        say(f"recovered {key}: marked aborted")
    store.set_section("notes", state.notes)

    rows = env.doctor(cfg)
    for row in rows:
        say(f"check {row.get('check')}: [{row.get('status')}] {row.get('text')}")
        if row.get("detail"):
            say(f"  {row['detail']}")
    store.set_section("doctor", _stored_doctor_rows(rows))
    for finding in findings:
        state.note(finding["note"])
    try:
        reset_check = guard.driver_reset_since(datetime.now(timezone.utc))
    except Exception:
        reset_check = None
    if reset_check is None:
        say("warning: GPU driver reset detection is unavailable (the Windows event log could "
            "not be read). A driver reset during the run would not be noticed.")
        state.note("GPU driver reset detection was unavailable during this run")

    aborted: RunAborted | None = None
    interrupted = False
    with env.keep_awake():
        try:
            for engine_name in cfg.engines:
                try:
                    _run_engine(state, engine_name)
                except (RunAborted, KeyboardInterrupt):
                    raise
                except Exception as err:
                    _retryable(state, f"{engine_name} stopped with an unexpected error")
                    state.problem(f"{engine_name}: stopped with an unexpected error: "
                                  f"{clean_error(err)}")
                    state.note(f"{engine_name} stopped with an unexpected error "
                               f"({clean_error(err)})")
        except RunAborted as err:
            aborted = err
        except KeyboardInterrupt:
            interrupted = True
            _mark_interrupted(state)

    if aborted is not None:
        say(f"Run ended early: {aborted}. The GPU driver was reset, so nothing more is "
            "started. Results so far are saved. Reboot the PC, then run the same command "
            "again to continue; the step that caused the reset is not repeated automatically.")
    if interrupted:
        say("Interrupted. Engines were stopped and results are saved. Run the same command "
            "again to continue.")

    report_failure = None
    try:
        report.write_report(run_dir)
        say(f"Report: {run_dir / 'README.md'}")
    except PrivacyError as err:
        report_failure = ("the report was not written, personal data was found. Fix "
                          f"results.json or raw/ by hand and run `bench report`: {err}")
    except (OSError, ValueError) as err:
        report_failure = f"the report was not written: {_clean(str(err))}"
    if report_failure:
        state.problems.append(report_failure)

    counts: dict[str, int] = {}
    for entry in store.data["tests"].values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    say(f"Results: {run_dir / 'results.json'}")
    say("Tests: " + (", ".join(f"{s} {n}" for s, n in sorted(counts.items())) or "none"))
    retryable = sorted(k for k, e in store.data["tests"].items()
                       if e["status"] in ("error", "running"))
    aborted_keys = sorted(k for k, e in store.data["tests"].items() if e["status"] == "aborted")
    if aborted_keys:
        say("Steps that caused a GPU driver reset (or were left running by a crash) are not "
            "repeated automatically. To measure one again, run with --force-test <key>:")
        for key in aborted_keys[:20]:
            say(f"  --force-test {key}")
    if retryable and not (aborted or interrupted):
        state.problems.append(f"{len(retryable)} test(s) ended with error or are unfinished")
        say(f"{len(retryable)} test(s) ended with an error or are unfinished; run the same "
            "command again to retry them.")

    if report_failure:
        say(f"problem: {report_failure}")  # last line: nothing may bury it
    if interrupted:
        return EXIT_INTERRUPTED
    if aborted is not None:
        return EXIT_ABORTED
    return EXIT_PROBLEMS if state.problems else EXIT_OK
