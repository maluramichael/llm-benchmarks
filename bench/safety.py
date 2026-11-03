"""Safety guard for the context ladder and the `bench doctor` checks.

Everything here is read-only: probes only look at the system, never change it.
Every external command has a timeout (gpu.run_text), and every probe failure
becomes "unknown" (None) or an `info`/`warn` row, never an exception.

Ladder keys: a context step is stored as `<key_prefix>ctx-<tokens>`.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from bench.gpu import run_text

IS_WINDOWS = sys.platform == "win32"
MIB = 1024 * 1024
RESERVE_MB = 1536
SMALL_CARD_MB = 4096
SMALL_CARD_RESERVE_MB = 512
MIN_GPU_FRACTION = 0.999  # anything below 100 % on the GPU aborts (rounding tolerance)
MAX_SHARED_GROWTH_MB = 500
MAX_SWAP_GROWTH_MB = 1024
UPTIME_WARN_MIN = 30
BIG_PROCESS_MB = 4096
TOP_PROCESSES = 15
PROBE_TIMEOUT_S = 40
WATCHDOG_DIR = Path(r"C:\Windows\LiveKernelReports\WATCHDOG")
EVENT_NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
_CTX_KEY = re.compile(r"ctx-(\d+)$")

ADAPTER_SCRIPT = (
    "Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage','\\GPU Adapter Memory(*)\\Dedicated Usage' "
    "-ErrorAction SilentlyContinue | ForEach-Object { $_.CounterSamples } | "
    "Select-Object InstanceName, Path, CookedValue | ConvertTo-Json -Compress")
PAGEFILE_SCRIPT = (
    "Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum | "
    "Select-Object -ExpandProperty Sum")
DOCTOR_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$tdr = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' |
  Select-Object TdrDelay, TdrDdiDelay, TdrLevel, HwSchMode
$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$result = [ordered]@{
  uptime_min = ((Get-Date) - $boot).TotalMinutes
  tdr = $tdr
  monitors = @(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBasicDisplayParams | Select-Object Active)
  processes = @(Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 ProcessName, WorkingSet64)
  pagefile = Get-CimInstance Win32_PageFileUsage | Select-Object -First 1 CurrentUsage, PeakUsage
}
$result | ConvertTo-Json -Depth 4 -Compress
""".strip()


@dataclass
class Reading:
    vram_free_mb: int | None
    gpu_fraction: float | None  # share of the model on the GPU, 1.0 = fully
    shared_mb: int | None  # shared/GTT usage of the GPU (absolute)
    swap_mb: int | None  # swap or pagefile usage (absolute)


def _list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _to_utc(ts: datetime) -> datetime:
    return ts.astimezone(timezone.utc)  # naive values are taken as local time


class Guard:
    def __init__(self, cfg, gpu_index: int, vram_total_mb: int):
        self.cfg = cfg
        self.gpu_index = gpu_index
        self.vram_total_mb = vram_total_mb

    def reserve_mb(self) -> int:
        if 0 < self.vram_total_mb <= SMALL_CARD_MB:
            return SMALL_CARD_RESERVE_MB
        return RESERVE_MB

    def before_step(self, required_mb: int, reading: Reading) -> tuple[bool, str]:
        if reading.vram_free_mb is None:
            return True, "free VRAM unknown, not checked"
        left = reading.vram_free_mb - required_mb
        reserve = self.reserve_mb()
        if left < reserve:
            return False, (f"only {left} MB would stay free ({reading.vram_free_mb} MB free, "
                           f"{required_mb} MB needed), reserve is {reserve} MB")
        return True, "enough VRAM free"

    def after_step(self, prev: Reading, now: Reading) -> tuple[bool, str]:
        if now.gpu_fraction is not None and now.gpu_fraction < MIN_GPU_FRACTION:
            return False, f"model only {now.gpu_fraction * 100:.0f} % on the GPU (partial offload)"
        if prev.shared_mb is not None and now.shared_mb is not None:
            grown = now.shared_mb - prev.shared_mb
            if grown > MAX_SHARED_GROWTH_MB:
                return False, f"shared GPU memory grew by {grown} MB (limit {MAX_SHARED_GROWTH_MB} MB)"
        if prev.swap_mb is not None and now.swap_mb is not None:
            grown = now.swap_mb - prev.swap_mb
            if grown > MAX_SWAP_GROWTH_MB:
                return False, f"swap grew by {grown} MB (limit {MAX_SWAP_GROWTH_MB} MB)"
        return True, "ok"

    def driver_reset_since(self, ts: datetime) -> bool:
        """True if Windows logged a GPU driver reset (TDR) after `ts`. False elsewhere or if unknown."""
        if not IS_WINDOWS:
            return False
        try:
            return _events_since(ts) or _watchdog_dump_since(ts)
        except Exception:
            return False


def _events_since(ts: datetime) -> bool:
    stamp = _to_utc(ts).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    query = ("*[System[((Provider[@Name='nvlddmkm'] and EventID=153) or "
             "(Provider[@Name='Display'] and EventID=4101)) and "
             f"TimeCreated[@SystemTime>='{stamp}']]]")
    out = run_text(["wevtutil", "qe", "System", f"/q:{query}", "/rd:true", "/f:xml"],
                   timeout=PROBE_TIMEOUT_S)
    return bool(parse_events(out))


def parse_events(text: str | None) -> list[tuple[str, int]]:
    """Parse wevtutil /f:xml output (events concatenated, no root) into (provider, id) pairs."""
    if not text or not text.strip():
        return []
    try:
        root = ET.fromstring("<Events>" + text.strip() + "</Events>")
    except ET.ParseError:
        return []
    found = []
    for system in root.iter(f"{EVENT_NS}System"):
        provider = system.find(f"{EVENT_NS}Provider")
        ident = system.find(f"{EVENT_NS}EventID")
        try:
            found.append((provider.get("Name", ""), int(ident.text)))
        except (AttributeError, TypeError, ValueError):
            continue
    return found


def _watchdog_dump_since(ts: datetime) -> bool:
    limit = _to_utc(ts).timestamp()
    try:
        return any(p.stat().st_mtime > limit for p in WATCHDOG_DIR.glob("*.dmp"))
    except OSError:
        return False


# --- readings ----------------------------------------------------------------------------

def parse_adapter_shared_mb(text: str | None) -> int | None:
    """Shared usage (MB) of the adapter with the largest dedicated usage (the real GPU)."""
    try:
        rows = [r for r in _list(json.loads(text or "")) if isinstance(r, dict)]
    except ValueError:
        return None
    shared: dict[str, float] = {}
    dedicated: dict[str, float] = {}
    for row in rows:
        name, path = row.get("InstanceName"), str(row.get("Path", "")).lower()
        value = row.get("CookedValue")
        if not name or not isinstance(value, (int, float)):
            continue
        if path.endswith("\\shared usage"):
            shared[name] = value
        elif path.endswith("\\dedicated usage"):
            dedicated[name] = value
    if not shared:
        return None
    pool = dedicated or shared
    best = max(pool, key=lambda n: pool[n])
    return int(shared.get(best, 0) // MIB)


def read_shared_mb() -> int | None:
    if not IS_WINDOWS:
        return None
    out = run_text(["powershell", "-NoProfile", "-NonInteractive", "-Command", ADAPTER_SCRIPT],
                   timeout=PROBE_TIMEOUT_S)
    return parse_adapter_shared_mb(out)


def read_swap_mb() -> int | None:
    try:
        if IS_WINDOWS:
            out = run_text(["powershell", "-NoProfile", "-NonInteractive", "-Command", PAGEFILE_SCRIPT],
                           timeout=PROBE_TIMEOUT_S)
            return int(float(out.strip())) if out and out.strip() else None
        if sys.platform.startswith("linux"):
            fields = dict(re.findall(r"^(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M))
            return (int(fields["SwapTotal"]) - int(fields["SwapFree"])) // 1024
    except (OSError, ValueError, KeyError):
        pass
    return None


def take_reading(gpu_index: int, gpu_fraction: float | None = None) -> Reading:
    """Read the current state. Unknown values are None. Never raises."""
    from bench.gpu import vram_free_mb
    try:
        free = vram_free_mb(gpu_index)
    except Exception:
        free = None
    return Reading(free, gpu_fraction, read_shared_mb(), read_swap_mb())


# --- context ladder ------------------------------------------------------------------------

def _normalise_steps(steps: Sequence[int] | None) -> list[int]:
    if steps is None:
        from bench.config import Config
        steps = Config().ctx_steps
    return sorted({int(s) for s in steps if int(s) > 0})


def ladder_start(store, key_prefix: str, steps: Sequence[int] | None = None) -> int | None:
    """First context step that may run, or None.

    None means: nothing left to run, or a step of this ladder is `aborted`,
    `does_not_fit` or was left `running` by a crash. In that case that step is
    recorded as `does_not_fit` and all higher steps without a final result are
    stored as `skipped` with a reason, so a second run cannot trigger another
    driver reset. `steps` default to the configured ctx_steps, normalised
    ascending and deduplicated.
    """
    ladder = _normalise_steps(steps)
    base = key_prefix + "ctx-"
    stored = {}
    for key, entry in store.data["tests"].items():
        if key.startswith(base) and _CTX_KEY.fullmatch(key[len(key_prefix):]):
            stored[int(key[len(base):])] = entry["status"]
    blockers = sorted(n for n, status in stored.items()
                      if status in ("aborted", "does_not_fit", "running"))
    if blockers:
        first = blockers[0]
        if stored[first] != "does_not_fit":
            store.save(base + str(first), "does_not_fit",
                       {"reason": f"step {first} crashed or was aborted in an earlier run"})
        for step in ladder:
            if step > first and stored.get(step) not in ("ok", "does_not_fit", "skipped"):
                store.save(base + str(step), "skipped",
                           {"reason": f"lower step {first} did not fit or crashed"})
        return None
    for step in ladder:
        if stored.get(step) not in ("ok", "skipped"):
            return step
    return None


# --- doctor --------------------------------------------------------------------------------

def parse_probe(text: str | None) -> dict | None:
    """Parse the JSON of DOCTOR_SCRIPT. PowerShell unwraps one-item arrays; undo that."""
    try:
        data = json.loads(text or "")
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    data["monitors"] = [m for m in _list(data.get("monitors")) if isinstance(m, dict)]
    data["processes"] = [p for p in _list(data.get("processes")) if isinstance(p, dict)]
    return data


def _windows_probe() -> dict | None:
    out = run_text(["powershell", "-NoProfile", "-NonInteractive", "-Command", DOCTOR_SCRIPT],
                   timeout=PROBE_TIMEOUT_S)
    return parse_probe(out)


def _row(check: str, status: str, text: str) -> dict:
    return {"check": check, "status": status, "text": text}


def doctor(cfg, platform: str | None = None,
           probe: Callable[[], dict | None] | None = None) -> list[dict]:
    """Read-only environment checks. Rows: {check, status: ok|warn|info, text}."""
    platform = platform or sys.platform
    names = ("uptime", "processes", "monitors", "tdr", "sysmem_fallback", "power_supply")
    if platform != "win32":
        return [_row(n, "info", "not available on this platform (Windows only)") for n in names]
    try:
        data = (probe or _windows_probe)()
    except Exception:
        data = None
    rows = []
    rows.append(_check_uptime(data))
    rows.append(_check_processes(data))
    rows.append(_check_monitors(data))
    rows.append(_check_tdr(data))
    rows.append(_row("sysmem_fallback", "info",
                     "NVIDIA 'CUDA - Sysmem Fallback Policy' can silently spill VRAM into system RAM "
                     "and distort results. Set it to 'Prefer No Sysmem Fallback' in the NVIDIA "
                     "control panel yourself; this tool does not change it."))
    rows.append(_row("power_supply", "info",
                     "Make sure the power supply has enough headroom for the GPU under load and all "
                     "PCIe power cables are connected; a weak supply can cause driver resets."))
    return rows


def _missing(check: str) -> dict:
    return _row(check, "info", "could not read this value (probe failed or timed out)")


def _check_uptime(data: dict | None) -> dict:
    value = (data or {}).get("uptime_min")
    if not isinstance(value, (int, float)):
        return _missing("uptime")
    minutes = int(value)
    if minutes > UPTIME_WARN_MIN:
        return _row("uptime", "warn", f"up for {minutes} min; a fresh boot gives cleaner results "
                                      f"(over {UPTIME_WARN_MIN} min)")
    return _row("uptime", "ok", f"up for {minutes} min")


def _check_processes(data: dict | None) -> dict:
    if not data or not data.get("processes"):
        return _missing("processes")
    procs = []
    for proc in data["processes"]:
        size = proc.get("WorkingSet64")
        if isinstance(size, (int, float)):
            procs.append((str(proc.get("ProcessName", "?")), int(size // MIB)))
    procs = sorted(procs, key=lambda p: -p[1])[:TOP_PROCESSES]
    if not procs:
        return _missing("processes")
    listing = ", ".join(f"{name} {mb} MB" for name, mb in procs)
    big = [name for name, mb in procs if mb >= BIG_PROCESS_MB]
    if big:
        return _row("processes", "warn", f"large processes ({', '.join(big)}); top {len(procs)}: {listing}")
    return _row("processes", "ok", f"top {len(procs)} by RAM: {listing}")


def _check_monitors(data: dict | None) -> dict:
    if data is None or "monitors" not in data:
        return _missing("monitors")
    active = sum(1 for m in data["monitors"] if m.get("Active"))
    return _row("monitors", "info", f"{active} active monitor(s); every monitor on the benchmark GPU "
                                    "uses some VRAM")


def _check_tdr(data: dict | None) -> dict:
    tdr = (data or {}).get("tdr")
    if data is None or "tdr" not in data:
        return _missing("tdr")
    tdr = tdr if isinstance(tdr, dict) else {}
    delay, ddi = tdr.get("TdrDelay"), tdr.get("TdrDdiDelay")
    parts = [f"TdrDelay {delay if delay is not None else 'not set (default 2 s)'}",
             f"TdrDdiDelay {ddi if ddi is not None else 'not set (default 5 s)'}"]
    text = ", ".join(parts) + ". Windows defaults reset the driver quickly under heavy load; " \
                              "this tool does not change them."
    return _row("tdr", "info", text)
