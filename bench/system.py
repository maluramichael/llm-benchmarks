"""System capture: OS, CPU, memory, GPUs, uptime, monitors, power, free disk.

Every probe is a small pure parser (tested with fixtures from real machines) plus
a thin runner that executes one command with a timeout and gives None on any
failure. collect_system() never raises: what cannot be read is None. It needs no
admin rights and changes nothing on the machine.

Keys beyond the documented ones: `model` (hardware model, e.g. "MacBook Pro") and
`laptop` (bool), both used for the device name suggestion.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from bench.gpu import GpuInfo, list_gpus, read_text, run_text

MIB = 1024 * 1024
WINDOWS_PROBE_TIMEOUT_S = 40  # PowerShell start plus CIM queries can be slow on a cold machine
DRM_ROOT = Path("/sys/class/drm")
POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")
# Virtual adapters that Windows lists next to the real GPUs.
VIRTUAL_GPU = re.compile(
    r"basic (render|display)|remote|virtual|mirror|parsec|citrix|vmware|hyper-v|displaylink"
    r"|indirect display|spacedesk", re.IGNORECASE)

KEYS = ("os", "os_version", "kernel", "cpu", "cores", "threads", "ram_gb", "ram_speed_mts",
        "gpus", "memory_for_models_gb", "uptime_min", "monitors_on_gpu", "power",
        "disk_free_gb", "model", "laptop")


# --- Small helpers ----------------------------------------------------------------------

def gb(n_bytes) -> int | float | None:
    """Bytes as GiB with one decimal, an int when whole. None for missing or zero."""
    try:
        value = float(n_bytes) / 2**30
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    value = round(value, 1)
    return int(value) if value.is_integer() else value


def clean_cpu_name(raw: str | None) -> str | None:
    if not raw:
        return None
    text = re.sub(r"\((?:R|TM)\)", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"\s*@\s*[\d.]+\s*[GM]Hz", "", text)
    text = re.sub(r"\b\d+-Core Processor\b", "", text)
    text = re.sub(r"\b(?:CPU|Processor)\b", "", text)
    return re.sub(r"\s+", " ", text).strip() or None


def vendor_of(name: str) -> str:
    lowered = name.lower()
    if any(word in lowered for word in ("nvidia", "geforce", "quadro", "rtx", "gtx")):
        return "nvidia"
    if any(word in lowered for word in ("amd", "radeon", "ati ")):
        return "amd"
    if "intel" in lowered:
        return "intel"
    if "apple" in lowered:
        return "apple"
    return "unknown"


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _dicts(value) -> list[dict]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _int(value) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


def _minutes_since(start: datetime | None, now: datetime | None) -> int | None:
    if start is None:
        return None
    if start.tzinfo is None:
        start = start.astimezone()
    now = now or datetime.now(timezone.utc)
    return max(0, int((now - start).total_seconds() // 60))


def _json(text: str | None):
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return None


# --- Linux ------------------------------------------------------------------------------

def parse_lscpu(text: str) -> dict:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    per_socket, sockets = _int(fields.get("Core(s) per socket")), _int(fields.get("Socket(s)"))
    return {"cpu": clean_cpu_name(fields.get("Model name")),
            "cores": per_socket * sockets if per_socket and sockets else None,
            "threads": _int(fields.get("CPU(s)"))}


def parse_meminfo_bytes(text: str) -> int | None:
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text or "", re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def parse_os_release(text: str) -> tuple[str | None, str | None]:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip().strip("\"'")
    name = fields.get("NAME") or fields.get("PRETTY_NAME") or None
    version = fields.get("VERSION") or fields.get("VERSION_ID") or None
    return name, version


def parse_proc_uptime(text: str) -> int | None:
    try:
        return int(float((text or "").split()[0]) // 60)
    except (IndexError, ValueError):
        return None


def read_power_supplies(root: Path = POWER_SUPPLY_ROOT) -> list[dict]:
    supplies: list[dict] = []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return supplies
    for entry in entries:
        supplies.append({key: (read_text(entry / key) or "").strip()
                         for key in ("type", "online", "status")})
    return supplies


def parse_power_supplies(supplies: list[dict]) -> tuple[bool, str]:
    """Return (has_battery, power) with power "ac" or "battery". No battery means desktop."""
    batteries = [s for s in supplies if s.get("type", "").lower() == "battery"]
    mains = [s for s in supplies if s.get("type", "").lower() in ("mains", "ups")]
    if not batteries:
        return False, "ac"
    if any(s.get("online") == "1" for s in mains):
        return True, "ac"
    if mains:
        return True, "battery"
    discharging = any(s.get("status", "").lower() == "discharging" for s in batteries)
    return True, "battery" if discharging else "ac"


def read_connector_statuses(root: Path = DRM_ROOT, card: str | None = None) -> list[str]:
    """Status strings of DRM connectors, e.g. 'connected' (optionally for one card)."""
    prefix = f"{card}-" if card else None
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    statuses: list[str] = []
    for entry in entries:
        if not re.fullmatch(r"card\d+-.+", entry.name):
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        text = read_text(entry / "status")
        if text is not None:
            statuses.append(text.strip())
    return statuses


def count_connected(statuses: list[str]) -> int:
    return sum(1 for status in statuses if status.strip() == "connected")


def _collect_linux(info: dict) -> None:
    name, version = parse_os_release(read_text("/etc/os-release") or "")
    info["os"], info["os_version"] = name, version
    kernel = (read_text("/proc/sys/kernel/osrelease") or "").strip()
    info["kernel"] = kernel or platform.release() or None
    lscpu = parse_lscpu(run_text(["lscpu"]) or "")
    info["cpu"], info["cores"] = lscpu["cpu"], lscpu["cores"]
    info["threads"] = lscpu["threads"] or os.cpu_count()
    info["ram_gb"] = gb(parse_meminfo_bytes(read_text("/proc/meminfo") or ""))
    info["uptime_min"] = parse_proc_uptime(read_text("/proc/uptime") or "")
    info["monitors_on_gpu"] = count_connected(read_connector_statuses())
    info["laptop"], info["power"] = parse_power_supplies(read_power_supplies())


# --- Windows ----------------------------------------------------------------------------

POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$result = [ordered]@{
  os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, LastBootUpTime, FreePhysicalMemory, TotalVisibleMemorySize
  cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
  mem = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, Speed, ConfiguredClockSpeed)
  gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion)
  monitors = @(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBasicDisplayParams | Select-Object Active)
  battery = @(Get-CimInstance Win32_Battery | Select-Object BatteryStatus)
  cs = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, PCSystemType, TotalPhysicalMemory
}
$result | ConvertTo-Json -Depth 4 -Compress
""".strip()


def powershell_command() -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_SCRIPT]


def parse_cim_datetime(value) -> datetime | None:
    """Parse '/Date(ms)/' (Windows PowerShell 5.1) or an ISO string (PowerShell 7)."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/", value.strip())
    if match:
        return datetime.fromtimestamp(int(match.group(1)) / 1000, tz=timezone.utc)
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_windows_cim(text: str, now: datetime | None = None) -> dict:
    """Turn the JSON of POWERSHELL_SCRIPT into system fields (PowerShell unwraps one-item arrays)."""
    data = _json(text)
    data = data if isinstance(data, dict) else {}
    os_info = data["os"] if isinstance(data.get("os"), dict) else {}
    cs = data["cs"] if isinstance(data.get("cs"), dict) else {}
    cpus, modules = _dicts(data.get("cpu")), _dicts(data.get("mem"))

    capacity = sum(_int(m.get("Capacity")) or 0 for m in modules)
    ram = (gb(capacity) or gb(cs.get("TotalPhysicalMemory"))
           or gb((_int(os_info.get("TotalVisibleMemorySize")) or 0) * 1024))
    speeds = [_int(m.get("ConfiguredClockSpeed")) or _int(m.get("Speed")) for m in modules]
    speeds = [s for s in speeds if s]
    batteries = _dicts(data.get("battery"))
    on_battery = any(_int(b.get("BatteryStatus")) in (1, 4, 5) for b in batteries)

    gpus: list[GpuInfo] = []
    for entry in _dicts(data.get("gpu")):
        name = str(entry.get("Name") or "").strip()
        if not name or VIRTUAL_GPU.search(name):
            continue
        gpus.append(GpuInfo(
            index=len(gpus), vendor=vendor_of(name), name=name,
            vram_total_mb=(_int(entry.get("AdapterRAM")) or 0) // MIB,
            vram_free_mb=None, driver=str(entry.get("DriverVersion") or "").strip() or None,
            backend=None))

    caption = str(os_info.get("Caption") or "").strip()
    return {
        "os": caption or None,
        "os_version": str(os_info.get("Version") or "").strip() or None,
        "cpu": clean_cpu_name(" ".join(str(c.get("Name") or "") for c in cpus[:1])),
        "cores": sum(_int(c.get("NumberOfCores")) or 0 for c in cpus) or None,
        "threads": sum(_int(c.get("NumberOfLogicalProcessors")) or 0 for c in cpus) or None,
        "ram_gb": ram,
        "ram_speed_mts": max(speeds) if speeds else None,
        "uptime_min": _minutes_since(parse_cim_datetime(os_info.get("LastBootUpTime")), now),
        "monitors_on_gpu": sum(1 for m in _dicts(data.get("monitors")) if m.get("Active") is True),
        "power": "battery" if on_battery else "ac",
        "laptop": bool(batteries) or _int(cs.get("PCSystemType")) == 2,
        "model": str(cs.get("Model") or "").strip() or None,
        "gpus": gpus,
    }


def _collect_windows(info: dict) -> list[GpuInfo]:
    parsed = parse_windows_cim(run_text(powershell_command(), timeout=WINDOWS_PROBE_TIMEOUT_S) or "")
    extra = parsed.pop("gpus", [])
    info.update({key: value for key, value in parsed.items() if value is not None})
    return extra


def _merge_windows_gpus(gpus: list[GpuInfo], cim: list[GpuInfo]) -> list[GpuInfo]:
    """Add what nvidia-smi cannot see (AMD, Intel). VRAM of those is capped at 4 GB by WMI."""
    merged = list(gpus)
    for item in cim:
        if item.vendor == "nvidia" and any(g.vendor == "nvidia" for g in merged):
            continue
        merged.append(replace(item, index=len(merged)))
    return merged


# --- macOS ------------------------------------------------------------------------------

def parse_mac_hardware(text: str) -> dict:
    data = _json(text)
    items = data.get("SPHardwareDataType") if isinstance(data, dict) else None
    entry = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
    memory = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(GB|TB)\s*", str(entry.get("physical_memory", "")))
    ram = None
    if memory:
        ram = float(memory.group(1)) * (1024 if memory.group(2) == "TB" else 1)
        ram = int(ram) if ram.is_integer() else ram
    return {"model": entry.get("machine_name") or None, "cpu": entry.get("chip_type") or None,
            "ram_gb": ram}


def parse_mac_displays(text: str) -> int | None:
    data = _json(text)
    adapters = data.get("SPDisplaysDataType") if isinstance(data, dict) else None
    if not isinstance(adapters, list):
        return None
    return sum(1 for adapter in adapters if isinstance(adapter, dict)
               for display in _dicts(adapter.get("spdisplays_ndrvs"))
               if display.get("spdisplays_online") == "spdisplays_yes")


def parse_pmset(text: str) -> str | None:
    if "'AC Power'" in (text or ""):
        return "ac"
    if "'Battery Power'" in (text or ""):
        return "battery"
    return None


def parse_boottime_uptime_min(text: str, now: datetime | None = None) -> int | None:
    match = re.search(r"sec\s*=\s*(\d+)", text or "")
    if not match:
        return None
    return _minutes_since(datetime.fromtimestamp(int(match.group(1)), tz=timezone.utc), now)


def _collect_macos(info: dict) -> None:
    def out(*args: str) -> str:
        return (run_text(list(args)) or "").strip()

    info["os"] = "macOS"
    info["os_version"] = out("sw_vers", "-productVersion") or None
    info["kernel"] = out("uname", "-r") or None
    hardware = parse_mac_hardware(out("system_profiler", "-json", "SPHardwareDataType"))
    info["model"] = hardware["model"]
    info["cpu"] = hardware["cpu"] or out("sysctl", "-n", "machdep.cpu.brand_string") or None
    info["ram_gb"] = hardware["ram_gb"] or gb(_int(out("sysctl", "-n", "hw.memsize")))
    info["cores"] = _int(out("sysctl", "-n", "hw.physicalcpu"))
    info["threads"] = _int(out("sysctl", "-n", "hw.logicalcpu"))
    info["uptime_min"] = parse_boottime_uptime_min(out("sysctl", "-n", "kern.boottime"))
    info["power"] = parse_pmset(out("pmset", "-g", "batt"))
    info["laptop"] = str(info["model"] or "").startswith("MacBook")
    info["monitors_on_gpu"] = parse_mac_displays(out("system_profiler", "-json", "SPDisplaysDataType"))


# --- Public interface -------------------------------------------------------------------

def _memory_for_models_gb(gpus: list[GpuInfo], ram_gb):
    apple = [g for g in gpus if g.vendor == "apple"]
    if apple:  # unified memory: only Metal's working set limit is usable, never the full RAM
        return gb(apple[0].vram_total_mb * MIB)
    sizes = [g.vram_total_mb for g in gpus if g.vram_total_mb > 0]
    return gb(max(sizes) * MIB) if sizes else ram_gb


def collect_system() -> dict:
    """Collect the facts about this machine. Never raises; unknown values are None."""
    info: dict = {key: None for key in KEYS}
    info["gpus"] = []
    cim_gpus: list[GpuInfo] = []

    def attempt(step, *args):
        try:
            return step(*args)
        except Exception:
            return None

    if sys.platform == "win32":
        cim_gpus = attempt(_collect_windows, info) or []
    elif sys.platform == "darwin":
        attempt(_collect_macos, info)
    elif sys.platform.startswith("linux"):
        attempt(_collect_linux, info)
    else:
        info["os"] = attempt(platform.system) or None
        info["threads"] = attempt(os.cpu_count)

    gpus = attempt(list_gpus) or []
    if cim_gpus:
        gpus = attempt(_merge_windows_gpus, gpus, cim_gpus) or gpus
    info["gpus"] = [asdict(g) for g in gpus]
    info["memory_for_models_gb"] = attempt(_memory_for_models_gb, gpus, info["ram_gb"])
    usage = attempt(shutil.disk_usage, Path.cwd())
    info["disk_free_gb"] = gb(usage.free) if usage else None
    return info


def _gpu_field(item, field: str):
    return item.get(field) if isinstance(item, dict) else getattr(item, field, None)


def _short_gpu_name(name: str) -> str:
    text = re.sub(r"\((?:R|TM)\)", "", name, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:NVIDIA|AMD)\s+", "", text)
    text = re.sub(r"\bGeForce\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _gb_label(value) -> str:
    return f" {round(value)} GB" if value else ""


def suggest_device_name(system: dict) -> str:
    """Suggest a device name like 'PC RTX 3090 24 GB' or 'MacBook Pro M5 Max 128 GB'."""
    cpu = system.get("cpu") or ""
    if cpu.startswith("Apple"):
        model = system.get("model") or "Mac"
        chip = cpu[len("Apple"):].strip()
        return f"{model} {chip}{_gb_label(system.get('ram_gb'))}".strip()
    prefix = "Laptop" if system.get("laptop") else "PC"
    gpus = [g for g in system.get("gpus") or [] if _gpu_field(g, "name")]
    if gpus:
        best = max(gpus, key=lambda g: _gpu_field(g, "vram_total_mb") or 0)
        vram = (_gpu_field(best, "vram_total_mb") or 0) / 1024
        return f"{prefix} {_short_gpu_name(_gpu_field(best, 'name'))}{_gb_label(vram)}"
    return f"{prefix} {cpu}{_gb_label(system.get('ram_gb'))}".strip()
