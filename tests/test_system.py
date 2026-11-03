import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from bench import system
from bench.gpu import GpuInfo
from bench.system import suggest_device_name

FIXTURES = Path(__file__).resolve().parent / "fixtures"

KEYS = ["os", "os_version", "kernel", "cpu", "cores", "threads", "ram_gb", "ram_speed_mts",
        "gpus", "memory_for_models_gb", "uptime_min", "monitors_on_gpu", "power",
        "disk_free_gb", "model", "laptop"]


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def gpu_dict(name, vram_mb, vendor="nvidia"):
    return {"index": 0, "vendor": vendor, "name": name, "vram_total_mb": vram_mb,
            "vram_free_mb": None, "driver": None, "backend": None}


class DeviceNameTest(unittest.TestCase):
    def test_device_name_pc(self):
        system_info = {"gpus": [gpu_dict("NVIDIA GeForce RTX 3080 Ti", 12288)]}
        self.assertEqual(suggest_device_name(system_info), "PC RTX 3080 Ti 12 GB")

    def test_device_name_mac(self):
        system_info = {"model": "MacBook Pro", "cpu": "Apple M5 Max", "ram_gb": 128, "gpus": []}
        self.assertEqual(suggest_device_name(system_info), "MacBook Pro M5 Max 128 GB")

    def test_device_name_mac_float_ram(self):
        system_info = {"model": "Mac mini", "cpu": "Apple M4 Pro", "ram_gb": 64.0}
        self.assertEqual(suggest_device_name(system_info), "Mac mini M4 Pro 64 GB")

    def test_device_name_amd(self):
        system_info = {"gpus": [gpu_dict("Radeon RX 560", 4096, "amd")]}
        self.assertEqual(suggest_device_name(system_info), "PC Radeon RX 560 4 GB")

    def test_device_name_laptop(self):
        system_info = {"laptop": True,
                       "gpus": [gpu_dict("NVIDIA GeForce RTX 4060 Laptop GPU", 8188)]}
        self.assertEqual(suggest_device_name(system_info), "Laptop RTX 4060 Laptop GPU 8 GB")

    def test_device_name_picks_largest_gpu(self):
        system_info = {"gpus": [gpu_dict("Intel(R) UHD Graphics 630", 1024, "intel"),
                                gpu_dict("NVIDIA GeForce GTX 1080 Ti", 11264)]}
        self.assertEqual(suggest_device_name(system_info), "PC GTX 1080 Ti 11 GB")

    def test_device_name_accepts_gpuinfo_objects(self):
        info = GpuInfo(0, "nvidia", "NVIDIA GeForce RTX 3090", 24576, None, None, "cuda")
        self.assertEqual(suggest_device_name({"gpus": [info]}), "PC RTX 3090 24 GB")

    def test_device_name_without_gpu(self):
        system_info = {"cpu": "AMD Ryzen 7 3700X", "ram_gb": 16.0, "gpus": []}
        self.assertEqual(suggest_device_name(system_info), "PC AMD Ryzen 7 3700X 16 GB")

    def test_device_name_with_nothing_known(self):
        self.assertEqual(suggest_device_name({}), "PC")


class HelpersTest(unittest.TestCase):
    def test_clean_cpu_name(self):
        cases = {
            "AMD Ryzen 7 3700X 8-Core Processor             ": "AMD Ryzen 7 3700X",
            "Intel(R) Core(TM) i3-8100 CPU @ 3.60GHz": "Intel Core i3-8100",
            "Apple M5 Max": "Apple M5 Max",
            "  12th Gen Intel(R) Core(TM) i7-12700K  ": "12th Gen Intel Core i7-12700K",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(system.clean_cpu_name(raw), expected)

    def test_gb(self):
        self.assertEqual(system.gb(137438953472), 128)
        self.assertEqual(system.gb(8057844 * 1024), 7.7)
        self.assertIsNone(system.gb(None))
        self.assertIsNone(system.gb(0))


class LinuxParserTest(unittest.TestCase):
    def test_parse_lscpu(self):
        self.assertEqual(system.parse_lscpu(fixture("linux-lscpu.txt")),
                         {"cpu": "Intel Core i3-8100", "cores": 4, "threads": 4})

    def test_parse_lscpu_multi_socket_and_smt(self):
        text = ("CPU(s):  64\nModel name:  AMD EPYC 7452 32-Core Processor\n"
                "Thread(s) per core:  2\nCore(s) per socket:  32\nSocket(s):  1\n")
        self.assertEqual(system.parse_lscpu(text)["cores"], 32)
        self.assertEqual(system.parse_lscpu(text)["threads"], 64)

    def test_parse_lscpu_garbage(self):
        self.assertEqual(system.parse_lscpu("nothing useful"),
                         {"cpu": None, "cores": None, "threads": None})

    def test_parse_meminfo(self):
        self.assertEqual(system.parse_meminfo_bytes(fixture("linux-meminfo.txt")), 8057844 * 1024)
        self.assertIsNone(system.parse_meminfo_bytes("garbage"))

    def test_parse_os_release(self):
        self.assertEqual(system.parse_os_release(fixture("linux-os-release.txt")),
                         ("Debian GNU/Linux", "13 (trixie)"))
        self.assertEqual(system.parse_os_release('NAME="Arch Linux"\nID=arch\n'),
                         ("Arch Linux", None))
        self.assertEqual(system.parse_os_release("PRETTY_NAME='Foo 1'\n"), ("Foo 1", None))
        self.assertEqual(system.parse_os_release(""), (None, None))

    def test_parse_proc_uptime(self):
        self.assertEqual(system.parse_proc_uptime(fixture("linux-uptime.txt")), 108904)
        self.assertEqual(system.parse_proc_uptime("90.5 10.0"), 1)
        self.assertIsNone(system.parse_proc_uptime(""))

    def test_parse_power_supplies(self):
        ac = {"type": "Mains", "online": "1"}
        battery = {"type": "Battery", "status": "Discharging"}
        self.assertEqual(system.parse_power_supplies([]), (False, "ac"))
        self.assertEqual(system.parse_power_supplies([ac, battery]), (True, "ac"))
        self.assertEqual(system.parse_power_supplies([{"type": "Mains", "online": "0"}, battery]),
                         (True, "battery"))
        self.assertEqual(system.parse_power_supplies([battery]), (True, "battery"))
        self.assertEqual(system.parse_power_supplies(
            [{"type": "Battery", "status": "Full"}]), (True, "ac"))

    def test_count_connected(self):
        self.assertEqual(system.count_connected(["connected", "disconnected", "connected\n"]), 2)
        self.assertEqual(system.count_connected([]), 0)

    def test_read_drm_status_and_power_from_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, status in [("card0-DP-1", "connected"), ("card0-HDMI-A-1", "disconnected"),
                                 ("card1-DP-2", "connected")]:
                (root / name).mkdir()
                (root / name / "status").write_text(status + "\n")
            (root / "card0").mkdir()
            self.assertEqual(system.read_connector_statuses(root, card="card0"), ["connected", "disconnected"])
            self.assertEqual(system.read_connector_statuses(root), ["connected", "disconnected", "connected"])
            self.assertEqual(system.read_connector_statuses(root / "missing"), [])
            supply = root / "ps"
            (supply / "BAT0").mkdir(parents=True)
            (supply / "BAT0" / "type").write_text("Battery\n")
            (supply / "BAT0" / "status").write_text("Discharging\n")
            (supply / "AC").mkdir()
            (supply / "AC" / "type").write_text("Mains\n")
            (supply / "AC" / "online").write_text("0\n")
            self.assertEqual(system.parse_power_supplies(system.read_power_supplies(supply)),
                             (True, "battery"))
            self.assertEqual(system.read_power_supplies(root / "missing"), [])


class WindowsParserTest(unittest.TestCase):
    NOW = datetime.fromtimestamp(1793656679.5 + 90 * 60, tz=timezone.utc)

    def test_parse_windows_cim_desktop(self):
        info = system.parse_windows_cim(fixture("win-system.json"), now=self.NOW)
        self.assertEqual(info["os"], "Microsoft Windows 11 IoT Enterprise LTSC")
        self.assertEqual(info["os_version"], "10.0.26100")
        self.assertEqual(info["cpu"], "AMD Ryzen 7 3700X")
        self.assertEqual(info["cores"], 8)
        self.assertEqual(info["threads"], 16)
        self.assertEqual(info["ram_gb"], 16)
        self.assertEqual(info["ram_speed_mts"], 3600)
        self.assertEqual(info["uptime_min"], 90)
        self.assertEqual(info["monitors_on_gpu"], 1)
        self.assertEqual(info["power"], "ac")
        self.assertEqual(info["laptop"], False)
        self.assertEqual(info["model"], "MS-7C92")
        self.assertEqual(len(info["gpus"]), 1)
        gpu = info["gpus"][0]
        self.assertEqual((gpu.vendor, gpu.name, gpu.driver), ("nvidia", "NVIDIA GeForce RTX 3080 Ti", "32.0.15.9159"))

    def test_parse_windows_cim_laptop_with_arrays_and_iso_date(self):
        now = datetime.fromisoformat("2026-11-02T10:15:30+01:00")
        info = system.parse_windows_cim(fixture("win-system-laptop.json"), now=now)
        self.assertEqual(info["cpu"], "Intel Core i7-13700H")
        self.assertEqual(info["ram_gb"], 32)
        self.assertEqual(info["ram_speed_mts"], 5200)
        self.assertEqual(info["uptime_min"], 120)
        self.assertEqual(info["monitors_on_gpu"], 2)
        self.assertEqual(info["power"], "battery")
        self.assertTrue(info["laptop"])
        self.assertEqual([g.name for g in info["gpus"]],
                         ["Intel(R) Iris(R) Xe Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"])
        self.assertEqual(info["gpus"][0].vram_total_mb, 1024)

    def test_parse_windows_cim_broken_input(self):
        for text in ["", "not json", "[]", "null", '{"os": 5, "cpu": "x"}']:
            with self.subTest(text=text):
                info = system.parse_windows_cim(text)
                self.assertIsInstance(info, dict)
                self.assertEqual(info.get("gpus", []), [])

    def test_parse_cim_datetime(self):
        expected = datetime.fromtimestamp(1793656679.5, tz=timezone.utc)
        self.assertEqual(system.parse_cim_datetime("/Date(1793656679500)/"), expected)
        self.assertEqual(system.parse_cim_datetime("2026-11-02T08:15:30+01:00"),
                         datetime.fromisoformat("2026-11-02T08:15:30+01:00"))
        self.assertIsNone(system.parse_cim_datetime("yesterday"))
        self.assertIsNone(system.parse_cim_datetime(None))

    def test_powershell_command_is_read_only_and_non_interactive(self):
        command = system.powershell_command()
        self.assertEqual(command[0].lower(), "powershell")
        self.assertIn("-NoProfile", command)
        self.assertIn("-NonInteractive", command)
        script = command[-1]
        self.assertIn("ConvertTo-Json", script)
        for cim in ["Win32_OperatingSystem", "Win32_Processor", "Win32_PhysicalMemory",
                    "Win32_VideoController", "WmiMonitorBasicDisplayParams", "Win32_Battery"]:
            self.assertIn(cim, script)
        self.assertNotIn("PNPDeviceID", script)
        self.assertNotRegex(script, r"\b(Set|Remove|New|Stop|Start|Invoke)-")


class MacParserTest(unittest.TestCase):
    def test_parse_mac_hardware(self):
        self.assertEqual(system.parse_mac_hardware(fixture("mac-hardware.json")),
                         {"model": "MacBook Pro", "cpu": "Apple M5 Max", "ram_gb": 128})

    def test_parse_mac_hardware_garbage(self):
        self.assertEqual(system.parse_mac_hardware("nope"),
                         {"model": None, "cpu": None, "ram_gb": None})

    def test_parse_mac_displays(self):
        self.assertEqual(system.parse_mac_displays(fixture("mac-displays.json")), 2)
        self.assertIsNone(system.parse_mac_displays("nope"))

    def test_parse_pmset(self):
        self.assertEqual(system.parse_pmset(fixture("mac-pmset-ac.txt")), "ac")
        self.assertEqual(system.parse_pmset(fixture("mac-pmset-battery.txt")), "battery")
        self.assertIsNone(system.parse_pmset(""))

    def test_parse_boottime(self):
        text = "{ sec = 1793524434, usec = 545978 } Sun Nov  1 10:13:54 2026"
        now = datetime.fromtimestamp(1793524434 + 61 * 60, tz=timezone.utc)
        self.assertEqual(system.parse_boottime_uptime_min(text, now=now), 61)
        self.assertIsNone(system.parse_boottime_uptime_min("garbage", now=now))


class CollectSystemTest(unittest.TestCase):
    def test_all_probes_failing_still_gives_every_key(self):
        with mock.patch.object(system, "run_text", return_value=None), \
                mock.patch.object(system, "list_gpus", return_value=[]), \
                mock.patch.object(system, "read_text", return_value=None):
            info = system.collect_system()
        self.assertEqual(sorted(info), sorted(KEYS))
        self.assertEqual(info["gpus"], [])
        json.dumps(info)

    def test_raising_probes_never_escape(self):
        boom = RuntimeError("boom")
        with mock.patch.object(system, "run_text", side_effect=boom), \
                mock.patch.object(system, "list_gpus", side_effect=boom), \
                mock.patch.object(system, "read_text", side_effect=boom), \
                mock.patch.object(system.shutil, "disk_usage", side_effect=boom):
            for platform in ["win32", "darwin", "linux", "freebsd"]:
                with self.subTest(platform=platform), \
                        mock.patch.object(system.sys, "platform", platform):
                    info = system.collect_system()
                    self.assertEqual(sorted(info), sorted(KEYS))

    def test_missing_working_directory_does_not_raise(self):
        with mock.patch.object(system, "run_text", return_value=None), \
                mock.patch.object(system, "list_gpus", return_value=[]), \
                mock.patch.object(system, "read_text", return_value=None), \
                mock.patch.object(system.Path, "cwd", side_effect=FileNotFoundError("gone")):
            info = system.collect_system()
        self.assertIsNone(info["disk_free_gb"])
        self.assertEqual(sorted(info), sorted(KEYS))

    def test_json_parser_strips_a_leading_bom(self):
        self.assertEqual(system._json("\ufeff" + '{"a": 1}'), {"a": 1})

    def test_linux_collection_from_fixtures(self):
        texts = {"/proc/cpuinfo": None,
                 "/proc/meminfo": fixture("linux-meminfo.txt"),
                 "/etc/os-release": fixture("linux-os-release.txt"),
                 "/proc/uptime": fixture("linux-uptime.txt"),
                 "/proc/sys/kernel/osrelease": fixture("linux-kernel.txt")}
        amd = GpuInfo(0, "amd", "Radeon RX 560", 4096, 4090, None, "vulkan")

        def run(args, **_kwargs):
            return fixture("linux-lscpu.txt") if args[0] == "lscpu" else None

        with mock.patch.object(system.sys, "platform", "linux"), \
                mock.patch.object(system, "run_text", side_effect=run), \
                mock.patch.object(system, "read_text", side_effect=lambda p: texts.get(str(p))), \
                mock.patch.object(system, "list_gpus", return_value=[amd]), \
                mock.patch.object(system, "read_power_supplies", return_value=[]), \
                mock.patch.object(system, "read_connector_statuses", return_value=[]):
            info = system.collect_system()
        self.assertEqual(info["os"], "Debian GNU/Linux")
        self.assertEqual(info["os_version"], "13 (trixie)")
        self.assertEqual(info["kernel"], "6.12.101+deb13-amd64")
        self.assertEqual(info["cpu"], "Intel Core i3-8100")
        self.assertEqual(info["cores"], 4)
        self.assertEqual(info["ram_gb"], 7.7)
        self.assertEqual(info["uptime_min"], 108904)
        self.assertEqual(info["monitors_on_gpu"], 0)
        self.assertEqual(info["power"], "ac")
        self.assertEqual(info["laptop"], False)
        self.assertEqual(info["memory_for_models_gb"], 4)
        self.assertEqual(info["gpus"][0]["name"], "Radeon RX 560")
        self.assertEqual(suggest_device_name(info), "PC Radeon RX 560 4 GB")

    def test_windows_collection_merges_cim_gpus_without_duplicating_nvidia(self):
        smi = [GpuInfo(0, "nvidia", "NVIDIA GeForce RTX 3080 Ti", 12288, 10514, "591.59", "cuda")]

        def run(args, **_kwargs):
            return fixture("win-system-laptop.json") if args[0].lower() == "powershell" else None

        with mock.patch.object(system.sys, "platform", "win32"), \
                mock.patch.object(system, "run_text", side_effect=run), \
                mock.patch.object(system, "list_gpus", return_value=smi):
            info = system.collect_system()
        self.assertEqual([g["vendor"] for g in info["gpus"]], ["nvidia", "intel"])
        self.assertEqual([g["index"] for g in info["gpus"]], [0, 1])
        self.assertEqual(info["memory_for_models_gb"], 12)
        self.assertEqual(info["laptop"], True)

    def test_mac_collection(self):
        apple = GpuInfo(0, "apple", "Apple M5 Max", 110100, None, None, "metal")
        outputs = {
            ("sw_vers", "-productVersion"): "26.6\n",
            ("uname", "-r"): "25.6.0\n",
            ("sysctl", "-n", "hw.physicalcpu"): "18\n",
            ("sysctl", "-n", "hw.logicalcpu"): "18\n",
            ("sysctl", "-n", "kern.boottime"): "{ sec = 1, usec = 2 } x",
            ("pmset", "-g", "batt"): fixture("mac-pmset-ac.txt"),
            ("system_profiler", "-json", "SPHardwareDataType"): fixture("mac-hardware.json"),
            ("system_profiler", "-json", "SPDisplaysDataType"): fixture("mac-displays.json"),
        }
        with mock.patch.object(system.sys, "platform", "darwin"), \
                mock.patch.object(system, "run_text", side_effect=lambda a, **k: outputs.get(tuple(a))), \
                mock.patch.object(system, "list_gpus", return_value=[apple]):
            info = system.collect_system()
        self.assertEqual(info["os"], "macOS")
        self.assertEqual(info["os_version"], "26.6")
        self.assertEqual(info["kernel"], "25.6.0")
        self.assertEqual(info["cpu"], "Apple M5 Max")
        self.assertEqual((info["cores"], info["threads"]), (18, 18))
        self.assertEqual(info["ram_gb"], 128)
        self.assertIsNone(info["ram_speed_mts"])
        self.assertEqual(info["memory_for_models_gb"], 107.5)
        self.assertEqual(info["monitors_on_gpu"], 2)
        self.assertEqual(info["power"], "ac")
        self.assertTrue(info["laptop"])
        self.assertEqual(suggest_device_name(info), "MacBook Pro M5 Max 128 GB")

    def test_real_collection_on_this_machine_is_complete_and_json(self):
        info = system.collect_system()
        self.assertEqual(sorted(info), sorted(KEYS))
        json.dumps(info)
        self.assertIsInstance(suggest_device_name(info), str)


if __name__ == "__main__":
    unittest.main()
