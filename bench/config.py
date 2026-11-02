"""Configuration: defaults, then bench.toml, then command line (last one wins)."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ENGINES = ("ollama", "llamacpp")
SUITES = ("speed", "memory", "smoke", "concurrency")
SCOPES = ("full", "no-context", "warm-only")
LLAMA_BACKENDS = ("cuda", "vulkan", "metal", "rocm", "cpu")
COMMANDS = {
    "system": "Collect and show system information, suggest a device name",
    "doctor": "Check prerequisites (uptime, open programs, monitors, driver settings)",
    "plan": "Check what fits into memory, estimate duration, mark missing combinations",
    "setup": "Find or fetch engines and models, measure nothing",
    "run": "Run the suites, saving after every single test",
    "report": "Render results.json to the run's README.md, with privacy check",
}
DEFAULT_CONFIG_FILE = "bench.toml"


class ConfigError(ValueError):
    """A parameter value is invalid."""


@dataclass
class Config:
    # Selection
    engines: list[str] = field(default_factory=lambda: ["ollama", "llamacpp"])
    models: list[str] = field(default_factory=lambda: ["S", "M", "L"])
    suites: list[str] = field(default_factory=lambda: ["speed", "memory", "smoke"])
    scope: str = "full"
    tasks: list[str] = field(default_factory=list)
    # Ollama
    ollama_url: str | None = None
    ollama_bin: Path | None = None
    ollama_models_dir: Path | None = None
    no_ollama_start: bool = False
    # llama.cpp
    llama_server: Path | None = None
    llama_bench: Path | None = None
    llama_release: str | None = None
    llama_backend: str | None = None
    llama_args: str = ""
    # Models
    gguf: dict[str, Path] = field(default_factory=dict)
    model_dir: Path | None = None
    mirror: str | None = None
    reuse_ollama_blobs: bool = False
    # Flow
    yes: bool = False
    dry_run: bool = False
    device: str | None = None
    gpu: int | None = None
    reps: int = 3
    ctx_steps: list[int] = field(
        default_factory=lambda: [8192, 16384, 32768, 65536, 131072, 262144])
    no_install: bool = False
    no_download: bool = False
    no_code_exec: bool = False
    thinking: bool = False
    output: Path = Path("runs")
    # Repeating
    force: bool = False
    force_tests: list[str] = field(default_factory=list)


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"expected a comma separated list, got {value!r}")
    items = [v.strip() for v in value if v.strip()]
    if not items:
        raise ConfigError("list must not be empty")
    return items


def _choice_list(allowed: tuple[str, ...], what: str):
    def convert(value) -> list[str]:
        items = _str_list(value)
        for item in items:
            if item not in allowed:
                raise ConfigError(
                    f"unknown {what} {item!r} (allowed: {', '.join(allowed)})")
        return items
    return convert


def _choice(allowed: tuple[str, ...], what: str):
    def convert(value) -> str:
        if value not in allowed:
            raise ConfigError(
                f"unknown {what} {value!r} (allowed: {', '.join(allowed)})")
        return value
    return convert


def _tokens(value) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"invalid token count {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip().lower()
        try:
            number = int(text[:-1]) * 1024 if text.endswith("k") else int(text)
        except ValueError:
            raise ConfigError(f"invalid token count {value!r} (examples: 4k, 8192)") from None
    else:
        raise ConfigError(f"invalid token count {value!r}")
    if number <= 0:
        raise ConfigError(f"token count must be positive, got {value!r}")
    return number


def _ctx_steps(value) -> list[int]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or not value:
        raise ConfigError(f"expected a comma separated list, got {value!r}")
    return [_tokens(v) for v in value]


def _int(minimum: int):
    def convert(value) -> int:
        if isinstance(value, bool):
            raise ConfigError(f"expected an integer, got {value!r}")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"expected an integer, got {value!r}") from None
        if number < minimum:
            raise ConfigError(f"must be at least {minimum}, got {number}")
        return number
    return convert


def _text(value) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"expected text, got {value!r}")
    return value


def _path(value) -> Path:
    return Path(_text(value))


def _bool(value) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"expected true or false, got {value!r}")
    return value


def _gguf(value) -> dict[str, Path]:
    if isinstance(value, dict):
        return {str(k): _path(v) for k, v in value.items()}
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ConfigError(f"expected <id>=<path> entries, got {value!r}")
    result: dict[str, Path] = {}
    for entry in value:
        model_id, sep, path = _text(entry).partition("=")
        if not sep or not model_id or not path:
            raise ConfigError(f"expected <id>=<path>, got {entry!r}")
        result[model_id] = Path(path)
    return result


# Converter per field, used for both bench.toml values and command line values.
CONVERTERS = {
    "engines": _choice_list(ENGINES, "engine"),
    "models": _str_list,
    "suites": _choice_list(SUITES, "suite"),
    "scope": _choice(SCOPES, "scope"),
    "tasks": _str_list,
    "ollama_url": _text,
    "ollama_bin": _path,
    "ollama_models_dir": _path,
    "no_ollama_start": _bool,
    "llama_server": _path,
    "llama_bench": _path,
    "llama_release": _text,
    "llama_backend": _choice(LLAMA_BACKENDS, "llama.cpp backend"),
    "llama_args": _text,
    "gguf": _gguf,
    "model_dir": _path,
    "mirror": _text,
    "reuse_ollama_blobs": _bool,
    "yes": _bool,
    "dry_run": _bool,
    "device": _text,
    "gpu": _int(0),
    "reps": _int(1),
    "ctx_steps": _ctx_steps,
    "no_install": _bool,
    "no_download": _bool,
    "no_code_exec": _bool,
    "thinking": _bool,
    "output": _path,
    "force": _bool,
    "force_tests": lambda value: [_text(v) for v in value] if isinstance(value, list)
    else [_text(value)],
}
assert set(CONVERTERS) == {f.name for f in fields(Config)}


def _option(name: str) -> str:
    return "--" + name.replace("_", "-")


def _common_options() -> argparse.ArgumentParser:
    """Options shared by every command. Unset options are absent from the namespace."""
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    add = common.add_argument
    add("--config", metavar="FILE",
        help=f"parameter file (default: ./{DEFAULT_CONFIG_FILE} if present)")

    sel = common.add_argument_group("selection")
    sel.add_argument(_option("engines"), metavar="LIST",
                     help=f"comma separated: {', '.join(ENGINES)} (default: all)")
    sel.add_argument(_option("models"), metavar="LIST",
                     help="comma separated classes or model ids (default: S,M,L)")
    sel.add_argument(_option("suites"), metavar="LIST",
                     help=f"comma separated: {', '.join(SUITES)} (default: speed,memory,smoke)")
    sel.add_argument(_option("scope"), metavar="SCOPE",
                     help=f"{' | '.join(SCOPES)} (default: full)")
    sel.add_argument(_option("tasks"), metavar="PATTERN",
                     help="comma separated smoke task patterns")

    oll = common.add_argument_group("ollama")
    oll.add_argument(_option("ollama_url"), metavar="URL",
                     help="use an already running server (never stopped by bench)")
    oll.add_argument(_option("ollama_bin"), metavar="PATH", help="ollama executable")
    oll.add_argument(_option("ollama_models_dir"), metavar="DIR",
                     help="OLLAMA_MODELS for the server bench starts")
    oll.add_argument(_option("no_ollama_start"), action="store_true",
                     default=argparse.SUPPRESS, help="do not start a server")

    lla = common.add_argument_group("llama.cpp")
    lla.add_argument(_option("llama_server"), metavar="PATH", help="llama-server executable")
    lla.add_argument(_option("llama_bench"), metavar="PATH", help="llama-bench executable")
    lla.add_argument(_option("llama_release"), metavar="TAG",
                     help="download this official release")
    lla.add_argument(_option("llama_backend"), metavar="BACKEND",
                     help=" | ".join(LLAMA_BACKENDS))
    lla.add_argument(_option("llama_args"), metavar="ARGS",
                     help="additional llama-server arguments")

    mod = common.add_argument_group("models")
    mod.add_argument(_option("gguf"), metavar="ID=PATH", action="append",
                     help="GGUF file for a model id (repeatable)")
    mod.add_argument(_option("model_dir"), metavar="DIR", help="directory with GGUF files")
    mod.add_argument(_option("mirror"), metavar="PATH_OR_URL", help="model mirror")
    mod.add_argument(_option("reuse_ollama_blobs"), action="store_true",
                     default=argparse.SUPPRESS,
                     help="use GGUF blobs from the Ollama store for llama.cpp")

    flow = common.add_argument_group("flow")
    flow.add_argument(_option("yes"), action="store_true", default=argparse.SUPPRESS,
                      help="no questions")
    flow.add_argument(_option("dry_run"), action="store_true", default=argparse.SUPPRESS,
                      help="show what would happen")
    flow.add_argument(_option("device"), metavar="NAME", help="device name")
    flow.add_argument(_option("gpu"), metavar="INDEX", help="GPU index")
    flow.add_argument(_option("reps"), metavar="N", help="repetitions (default: 3)")
    flow.add_argument(_option("ctx_steps"), metavar="LIST",
                      help="context ladder, e.g. 4k,8k,128k (default: 8k to 256k)")
    flow.add_argument(_option("no_install"), action="store_true", default=argparse.SUPPRESS,
                      help="do not install engines")
    flow.add_argument(_option("no_download"), action="store_true", default=argparse.SUPPRESS,
                      help="do not download models")
    flow.add_argument(_option("no_code_exec"), action="store_true", default=argparse.SUPPRESS,
                      help="skip smoke tasks that run model generated code")
    flow.add_argument(_option("thinking"), action="store_true", default=argparse.SUPPRESS,
                      help="allow thinking mode")
    flow.add_argument(_option("output"), metavar="DIR", help="output directory (default: runs)")

    rep = common.add_argument_group("repeating")
    rep.add_argument(_option("force"), action="store_true", default=argparse.SUPPRESS,
                     help="measure everything again")
    rep.add_argument(_option("force_test"), dest="force_tests", metavar="PATTERN",
                     action="append",
                     help="measure matching tests again, e.g. llamacpp/S/speed/* (repeatable)")
    return common


def build_parser() -> argparse.ArgumentParser:
    from bench import VERSION

    parser = argparse.ArgumentParser(
        prog="bench", description="Benchmark harness for local LLMs.")
    parser.add_argument("--version", action="version", version=f"bench {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    common = _common_options()
    for name, summary in COMMANDS.items():
        sub.add_parser(name, help=summary, description=summary, parents=[common])
    return parser


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as err:
        raise ConfigError(f"cannot read {path}: {err.strerror}") from None
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(f"invalid TOML in {path}: {err}") from None


def _apply(config: Config, values: dict, origin: str) -> None:
    for key, raw in values.items():
        if key not in CONVERTERS:
            raise ConfigError(f"unknown parameter {key!r} in {origin}")
        try:
            setattr(config, key, CONVERTERS[key](raw))
        except ConfigError as err:
            raise ConfigError(f"{_option(key)} ({origin}): {err}") from None


def load_config(argv: list[str]) -> tuple[str, Config]:
    """Parse argv. Precedence: defaults, bench.toml (or --config), command line.

    Invalid input ends the process with exit code 2 and a message on stderr.
    """
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    config_arg = args.pop("config", None)

    config = Config()
    try:
        if config_arg is not None:
            _apply(config, _read_toml(Path(config_arg)), config_arg)
        elif Path(DEFAULT_CONFIG_FILE).is_file():
            _apply(config, _read_toml(Path(DEFAULT_CONFIG_FILE)), DEFAULT_CONFIG_FILE)
        _apply(config, args, "command line")
    except ConfigError as err:
        parser.error(str(err))
    return command, config
