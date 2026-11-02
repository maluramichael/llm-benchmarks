"""GPU list and free VRAM per vendor.

Every probe is a small parser for command output (testable with fixtures) plus a
thin runner that executes the command and gives None on any failure. Nothing here
raises, needs admin rights or changes the system.

GpuInfo.index is one flat numbering: NVIDIA cards first (nvidia-smi order), then
AMD cards (sysfs order), then the Apple GPU. vram_total_mb 0 means unknown.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROBE_TIMEOUT_S = 15
MIB = 1024 * 1024
NVIDIA_QUERY = "index,name,memory.total,memory.used,memory.free,driver_version"
WINDOWS_NVIDIA_SMI = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)
AMD_SYSFS_ROOT = Path("/sys/class/drm")
AMD_PCI_VENDOR = "0x1002"
METAL_SWIFT = ("import Metal; "
               "print(MTLCreateSystemDefaultDevice()!.recommendedMaxWorkingSetSize / 1048576)")
GPU_PCI_CLASSES = ("VGA compatible controller", "3D controller", "Display controller")


@dataclass
class GpuInfo:
    index: int
    vendor: str  # nvidia, amd, intel, apple, unknown
    name: str
    vram_total_mb: int  # 0 when unknown
    vram_free_mb: int | None = None
    driver: str | None = None
    backend: str | None = None  # best llama.cpp backend: cuda, rocm, vulkan, metal


def run_text(args: list[str], timeout: float = PROBE_TIMEOUT_S,
             input_text: str | None = None) -> str | None:
    """Run a command and return its stdout, or None on any failure or timeout."""
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False, input=input_text,
            stdin=None if input_text is not None else subprocess.DEVNULL, **kwargs)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def read_text(path: str | Path) -> str | None:
    """Read a small text file (sysfs, procfs), or None when it is missing."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _int(text: str | None) -> int | None:
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


# --- NVIDIA ---------------------------------------------------------------------------

def parse_nvidia_smi(csv_text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=NVIDIA_QUERY --format=csv,noheader,nounits`."""
    gpus: list[GpuInfo] = []
    for line in (csv_text or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        index = _int(parts[0])
        total = _int(parts[-4])
        name = ", ".join(parts[1:-4])
        if index is None or total is None or not name:
            continue
        driver = parts[-1]
        gpus.append(GpuInfo(
            index=index, vendor="nvidia", name=name, vram_total_mb=total,
            vram_free_mb=_int(parts[-2]),
            driver=driver if driver and not driver.startswith("[") else None,
            backend="cuda"))
    return gpus


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform == "win32":
        for candidate in WINDOWS_NVIDIA_SMI:
            if Path(candidate).is_file():
                return candidate
    return None


def query_nvidia() -> list[GpuInfo]:
    exe = find_nvidia_smi()
    if not exe:
        return []
    text = run_text([exe, f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"])
    return parse_nvidia_smi(text) if text else []


# --- AMD ------------------------------------------------------------------------------

def simplify_marketing_name(raw: str) -> str:
    """Pick one card from a PCI name like 'Radeon RX 550 640SP / RX 560/560X'.

    The last alternative is used and a trailing '/560X' variant is dropped, so
    the example gives 'Radeon RX 560'. PCI ids cannot tell the cards of one chip
    apart, so this is a best guess (the user can override the device name).
    """
    alternatives = [part.strip() for part in raw.split(" / ") if part.strip()]
    if len(alternatives) < 2:
        return raw.strip()
    first_words = re.match(r"\D*", alternatives[0]).group(0).split()
    last_words = alternatives[-1].split("/")[0].split()
    for overlap in range(min(len(first_words), len(last_words)), -1, -1):
        if overlap == 0 or first_words[-overlap:] == last_words[:overlap]:
            return " ".join(first_words[:len(first_words) - overlap] + last_words)
    return alternatives[-1]


def parse_lspci_gpus(text: str) -> dict[str, str]:
    """Map PCI slot (e.g. '01:00.0') to the marketing name of display devices in `lspci`."""
    names: dict[str, str] = {}
    for line in (text or "").splitlines():
        match = re.match(r"^(\S+)\s+(.+?):\s+(.*)$", line)
        if not match or match.group(2) not in GPU_PCI_CLASSES:
            continue
        description = re.sub(r"\s*\(rev [0-9a-fA-F]+\)\s*$", "", match.group(3)).strip()
        bracket = re.search(r"\[([^\[\]]+)\]$", description)
        names[match.group(1)] = simplify_marketing_name(
            bracket.group(1) if bracket else description)
    return names


def amd_backend() -> str:
    return "rocm" if shutil.which("rocm-smi") else "vulkan"


def read_amd_sysfs(root: Path | None = None, names: dict[str, str] | None = None,
                   backend: str | None = None) -> list[GpuInfo]:
    """AMD cards from /sys/class/drm/card*/device/mem_info_vram_{total,used}."""
    root = AMD_SYSFS_ROOT if root is None else root
    names = names or {}
    try:
        cards = sorted((p for p in root.iterdir() if re.fullmatch(r"card\d+", p.name)),
                       key=lambda p: int(p.name[4:]))
    except OSError:
        return []
    gpus: list[GpuInfo] = []
    for card in cards:
        device = card / "device"
        vendor = (read_text(device / "vendor") or "").strip().lower()
        total = _int(read_text(device / "mem_info_vram_total"))
        used = _int(read_text(device / "mem_info_vram_used"))
        if vendor != AMD_PCI_VENDOR or total is None:
            continue
        slot = ""
        for line in (read_text(device / "uevent") or "").splitlines():
            if line.startswith("PCI_SLOT_NAME="):
                slot = line.split("=", 1)[1].strip()
        if slot.count(":") == 2:
            slot = slot.split(":", 1)[1]  # drop the PCI domain
        free = None if used is None else max(0, (total - used) // MIB)
        gpus.append(GpuInfo(
            index=len(gpus), vendor="amd", name=names.get(slot, f"AMD GPU ({card.name})"),
            vram_total_mb=total // MIB, vram_free_mb=free, driver=None, backend=backend))
    return gpus


def parse_rocm_smi_meminfo(text: str) -> list[GpuInfo]:
    """Parse `rocm-smi --showmeminfo vram --json` (fallback when sysfs gives nothing)."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    gpus: list[GpuInfo] = []
    for key in sorted((k for k in data if re.fullmatch(r"card\d+", k)), key=lambda k: int(k[4:])):
        entry = data[key]
        if not isinstance(entry, dict):
            continue
        total = _int(entry.get("VRAM Total Memory (B)"))
        used = _int(entry.get("VRAM Total Used Memory (B)"))
        if total is None:
            continue
        gpus.append(GpuInfo(
            index=len(gpus), vendor="amd", name=f"AMD GPU ({key})", vram_total_mb=total // MIB,
            vram_free_mb=None if used is None else max(0, (total - used) // MIB),
            driver=None, backend="rocm"))
    return gpus


def query_amd() -> list[GpuInfo]:
    names = parse_lspci_gpus(run_text(["lspci"]) or "")
    gpus = read_amd_sysfs(AMD_SYSFS_ROOT, names=names, backend=amd_backend())
    if not gpus and shutil.which("rocm-smi"):
        gpus = parse_rocm_smi_meminfo(
            run_text(["rocm-smi", "--showmeminfo", "vram", "--json"]) or "")
    return gpus


# --- Apple ----------------------------------------------------------------------------

def parse_swift_mib(text: str | None) -> int | None:
    """Parse the MiB number printed by METAL_SWIFT."""
    return _int(text.strip().splitlines()[-1]) if text and text.strip() else None


def metal_limit_mb() -> int | None:
    """Metal's recommendedMaxWorkingSetSize in MiB (the real model memory limit on a Mac)."""
    return parse_swift_mib(run_text(["swift", "-"], timeout=30, input_text=METAL_SWIFT))


def query_apple() -> list[GpuInfo]:
    name = (run_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()
    if not name.startswith("Apple"):
        return []  # Intel Macs with discrete GPUs are not supported in v1
    return [GpuInfo(index=0, vendor="apple", name=name, vram_total_mb=metal_limit_mb() or 0,
                    vram_free_mb=None, driver=None, backend="metal")]


# --- Public interface -----------------------------------------------------------------

def list_gpus() -> list[GpuInfo]:
    """All GPUs this module can see, in one flat numbering. Never raises."""
    gpus: list[GpuInfo] = []
    probes = [query_nvidia]
    if sys.platform.startswith("linux"):
        probes.append(query_amd)
    if sys.platform == "darwin":
        probes.append(query_apple)
    for probe in probes:
        try:
            found = probe()
        except Exception:
            continue
        for item in found:
            gpus.append(replace(item, index=len(gpus)) if gpus else item)
    return gpus


def vram_free_mb(index: int) -> int | None:
    """Free VRAM of one GPU in MiB: one nvidia-smi call or a sysfs read. Never raises."""
    try:
        if index < 0:
            return None
        nvidia = query_nvidia()
        for item in nvidia:
            if item.index == index:
                return item.vram_free_mb
        position = index - len(nvidia)
        amd = read_amd_sysfs(AMD_SYSFS_ROOT)
        if 0 <= position < len(amd):
            return amd[position].vram_free_mb
    except Exception:
        pass
    return None
