import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import gpu
from bench.gpu import GpuInfo, parse_nvidia_smi

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NVIDIA_CSV = (FIXTURES / "nvidia-smi.csv").read_text(encoding="utf-8")


class ParseNvidiaSmiTest(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        gpus = parse_nvidia_smi(NVIDIA_CSV)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0], GpuInfo(
            index=0, vendor="nvidia", name="NVIDIA GeForce RTX 3090",
            vram_total_mb=24576, vram_free_mb=22880, driver="591.59", backend="cuda"))
        self.assertEqual(gpus[1].name, "NVIDIA GeForce RTX 3080 Ti")
        self.assertEqual(gpus[1].vram_total_mb, 12288)
        self.assertEqual(gpus[1].vram_free_mb, 10514)
        self.assertEqual(gpus[1].index, 1)

    def test_real_single_line(self):
        gpus = parse_nvidia_smi("0, NVIDIA GeForce RTX 3080 Ti, 12288, 1571, 10514, 591.59\n")
        self.assertEqual([g.vram_total_mb for g in gpus], [12288])

    def test_not_available_fields_become_none(self):
        gpus = parse_nvidia_smi("0, NVIDIA Thing, 8192, [N/A], [Not Supported], [N/A]\n")
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vram_total_mb, 8192)
        self.assertIsNone(gpus[0].vram_free_mb)
        self.assertIsNone(gpus[0].driver)

    def test_name_with_comma(self):
        gpus = parse_nvidia_smi("0, NVIDIA Foo, Bar, 8192, 100, 8000, 570.1\n")
        self.assertEqual(gpus[0].name, "NVIDIA Foo, Bar")
        self.assertEqual(gpus[0].vram_total_mb, 8192)

    def test_garbage_and_empty_give_empty_list(self):
        for text in ["", "\n\n", "NVIDIA-SMI has failed because it couldn't communicate",
                     "x, y\n", "a, b, c, d, e, f\n"]:
            with self.subTest(text=text):
                self.assertEqual(parse_nvidia_smi(text), [])

    def test_windows_line_endings(self):
        gpus = parse_nvidia_smi(NVIDIA_CSV.replace("\n", "\r\n"))
        self.assertEqual(gpus[1].driver, "591.59")


class RunTextTest(unittest.TestCase):
    def test_returns_stdout(self):
        done = subprocess.CompletedProcess(["x"], 0, stdout="hello\n", stderr="")
        with mock.patch.object(gpu.subprocess, "run", return_value=done) as run:
            self.assertEqual(gpu.run_text(["x"], timeout=3), "hello\n")
        self.assertEqual(run.call_args.kwargs["timeout"], 3)

    def test_none_on_nonzero_exit(self):
        done = subprocess.CompletedProcess(["x"], 9, stdout="out", stderr="err")
        with mock.patch.object(gpu.subprocess, "run", return_value=done):
            self.assertIsNone(gpu.run_text(["x"]))

    def test_none_on_failure_modes(self):
        for error in [FileNotFoundError("x"), PermissionError("x"), OSError("x"),
                      subprocess.TimeoutExpired("x", 1), ValueError("embedded null")]:
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(gpu.subprocess, "run", side_effect=error):
                    self.assertIsNone(gpu.run_text(["x"]))

    def test_real_missing_binary(self):
        self.assertIsNone(gpu.run_text(["definitely-not-an-installed-program-xyz"]))

    def test_passes_input(self):
        done = subprocess.CompletedProcess(["x"], 0, stdout="", stderr="")
        with mock.patch.object(gpu.subprocess, "run", return_value=done) as run:
            gpu.run_text(["x"], input_text="abc")
        self.assertEqual(run.call_args.kwargs["input"], "abc")


class LspciTest(unittest.TestCase):
    def test_parse_lspci_gpus(self):
        text = (FIXTURES / "linux-lspci.txt").read_text(encoding="utf-8")
        self.assertEqual(gpu.parse_lspci_gpus(text), {"01:00.0": "Radeon RX 560"})

    def test_other_devices_are_ignored(self):
        text = ("00:1f.3 Audio device: Intel Corporation Sunrise Point-H HD Audio\n"
                "03:00.0 3D controller: NVIDIA Corporation GA102 [GeForce RTX 3090] (rev a1)\n")
        self.assertEqual(gpu.parse_lspci_gpus(text), {"03:00.0": "GeForce RTX 3090"})

    def test_simplify_amd_name(self):
        cases = {
            "Radeon RX 550 640SP / RX 560/560X": "Radeon RX 560",
            "Radeon RX 580 2048SP": "Radeon RX 580 2048SP",
            "Radeon RX 7900 XT": "Radeon RX 7900 XT",
            "Radeon RX 6800/6800 XT / 6900 XT": "Radeon RX 6900 XT",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(gpu.simplify_marketing_name(raw), expected)


class AmdSysfsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def card(self, name, vendor="0x1002", total=4294967296, used=5439488, slot="0000:01:00.0"):
        device = self.root / name / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
        (device / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={slot}\n")
        if total is not None:
            (device / "mem_info_vram_total").write_text(f"{total}\n")
        if used is not None:
            (device / "mem_info_vram_used").write_text(f"{used}\n")

    def test_reads_amd_card(self):
        self.card("card0")
        gpus = gpu.read_amd_sysfs(self.root, names={"01:00.0": "Radeon RX 560"}, backend="vulkan")
        self.assertEqual(gpus, [GpuInfo(
            index=0, vendor="amd", name="Radeon RX 560", vram_total_mb=4096,
            vram_free_mb=4090, driver=None, backend="vulkan")])

    def test_skips_connectors_other_vendors_and_incomplete_cards(self):
        self.card("card0")
        (self.root / "card0-DP-1").mkdir()
        (self.root / "renderD128").mkdir()
        self.card("card1", vendor="0x10de", slot="0000:02:00.0")
        self.card("card2", total=None, slot="0000:03:00.0")
        gpus = gpu.read_amd_sysfs(self.root, names={}, backend="vulkan")
        self.assertEqual([g.name for g in gpus], ["AMD GPU (card0)"])

    def test_cards_sorted_numerically(self):
        self.card("card10", slot="0000:0a:00.0")
        self.card("card2", slot="0000:02:00.0")
        gpus = gpu.read_amd_sysfs(self.root, names={}, backend="vulkan")
        self.assertEqual([g.index for g in gpus], [0, 1])
        self.assertEqual([g.name for g in gpus], ["AMD GPU (card2)", "AMD GPU (card10)"])

    def test_missing_root_gives_empty_list(self):
        self.assertEqual(gpu.read_amd_sysfs(self.root / "nope", names={}, backend="vulkan"), [])

    def test_free_never_negative(self):
        self.card("card0", total=1048576, used=9999999999)
        self.assertEqual(gpu.read_amd_sysfs(self.root, names={}, backend="vulkan")[0].vram_free_mb, 0)


class RocmTest(unittest.TestCase):
    def test_parse_rocm_smi_meminfo(self):
        text = ('{"card0": {"VRAM Total Memory (B)": "17163091968", '
                '"VRAM Total Used Memory (B)": "283508736"}, '
                '"card1": {"VRAM Total Memory (B)": "8589934592", '
                '"VRAM Total Used Memory (B)": "0"}}')
        gpus = gpu.parse_rocm_smi_meminfo(text)
        self.assertEqual([g.vram_total_mb for g in gpus], [16368, 8192])
        self.assertEqual(gpus[0].vram_free_mb, 16097)
        self.assertEqual(gpus[0].backend, "rocm")
        self.assertEqual(gpus[0].vendor, "amd")

    def test_garbage_gives_empty_list(self):
        for text in ["", "not json", "[]", '{"card0": "x"}', '{"system": {"a": "b"}}']:
            with self.subTest(text=text):
                self.assertEqual(gpu.parse_rocm_smi_meminfo(text), [])


class MetalTest(unittest.TestCase):
    def test_parse_swift_mib(self):
        self.assertEqual(gpu.parse_swift_mib("110100\n"), 110100)
        self.assertEqual(gpu.parse_swift_mib("110100.5\n"), 110100)
        self.assertIsNone(gpu.parse_swift_mib("error: no such module 'Metal'"))
        self.assertIsNone(gpu.parse_swift_mib(""))
        self.assertIsNone(gpu.parse_swift_mib(None))

    def test_metal_limit_runs_swift_with_stdin(self):
        with mock.patch.object(gpu, "run_text", return_value="110100\n") as run:
            self.assertEqual(gpu.metal_limit_mb(), 110100)
        args = run.call_args.args[0]
        self.assertEqual(args[0], "swift")
        self.assertIn("recommendedMaxWorkingSetSize", run.call_args.kwargs["input_text"])


class ListAndFreeTest(unittest.TestCase):
    def test_list_gpus_nvidia(self):
        with mock.patch.object(gpu, "sys", mock.Mock(platform="win32")), \
                mock.patch.object(gpu, "run_text", return_value=NVIDIA_CSV), \
                mock.patch.object(gpu.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(gpu, "AMD_SYSFS_ROOT", Path("/nonexistent-drm")):
            gpus = gpu.list_gpus()
        self.assertEqual([g.name for g in gpus],
                         ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3080 Ti"])

    def test_list_gpus_nothing_available(self):
        with mock.patch.object(gpu, "run_text", return_value=None), \
                mock.patch.object(gpu.shutil, "which", return_value=None), \
                mock.patch.object(gpu, "AMD_SYSFS_ROOT", Path("/nonexistent-drm")):
            self.assertEqual(gpu.list_gpus(), [])

    def test_list_gpus_never_raises(self):
        with mock.patch.object(gpu, "run_text", side_effect=RuntimeError("boom")), \
                mock.patch.object(gpu.shutil, "which", side_effect=RuntimeError("boom")):
            self.assertEqual(gpu.list_gpus(), [])

    def test_vram_free_mb_nvidia(self):
        with mock.patch.object(gpu, "run_text", return_value=NVIDIA_CSV) as run, \
                mock.patch.object(gpu.shutil, "which", return_value="nvidia-smi"):
            self.assertEqual(gpu.vram_free_mb(1), 10514)
            self.assertEqual(gpu.vram_free_mb(0), 22880)
        self.assertEqual(run.call_count, 2)  # one nvidia-smi call per query, nothing else

    def test_vram_free_mb_unknown_index_is_none(self):
        with mock.patch.object(gpu, "run_text", return_value=NVIDIA_CSV), \
                mock.patch.object(gpu.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(gpu, "AMD_SYSFS_ROOT", Path("/nonexistent-drm")):
            self.assertIsNone(gpu.vram_free_mb(7))
            self.assertIsNone(gpu.vram_free_mb(-1))

    def test_vram_free_mb_amd_sysfs_after_nvidia(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = Path(tmp) / "card0" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x1002\n")
            (device / "mem_info_vram_total").write_text("4294967296\n")
            (device / "mem_info_vram_used").write_text("5439488\n")
            with mock.patch.object(gpu, "run_text", return_value=None), \
                    mock.patch.object(gpu.shutil, "which", return_value=None), \
                    mock.patch.object(gpu, "AMD_SYSFS_ROOT", Path(tmp)):
                self.assertEqual(gpu.vram_free_mb(0), 4090)
                self.assertIsNone(gpu.vram_free_mb(1))

    def test_vram_free_mb_never_raises(self):
        with mock.patch.object(gpu, "run_text", side_effect=RuntimeError("boom")):
            self.assertIsNone(gpu.vram_free_mb(0))

    def test_find_nvidia_smi_windows_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "nvidia-smi.exe"
            exe.write_text("")
            with mock.patch.object(gpu.shutil, "which", return_value=None), \
                    mock.patch.object(gpu, "sys", mock.Mock(platform="win32")), \
                    mock.patch.object(gpu, "WINDOWS_NVIDIA_SMI", (str(exe),)):
                self.assertEqual(gpu.find_nvidia_smi(), str(exe))


if __name__ == "__main__":
    unittest.main()
