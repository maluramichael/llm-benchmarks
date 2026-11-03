import os
import unittest
from unittest import mock

from bench import preflight
from bench.config import Config

HEADER = """Mon Nov  3 10:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.59                 Driver Version: 591.59         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090      WDDM  |   00000000:01:00.0  On |                  N/A |
+-----------------------------------------+------------------------+----------------------+
"""
PROCESS_HEAD = """
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
"""
FOOT = "+-----------------------------------------------------------------------------------------+\n"


def table(*rows):
    return HEADER + PROCESS_HEAD + "".join(rows) + FOOT


def row(gpu, pid, kind, name, mem="N/A"):
    return (f"|{gpu:>5}   N/A  N/A {pid:>14} {kind:>6}   {name:<38}{mem:>10}  |\n")


# WDDM: ordinary desktop programs are graphics contexts (G or C+G).
DESKTOP = table(
    row(0, 1100, "C+G", r"C:\Windows\explorer.exe"),
    row(0, 1200, "C+G", r"C:\Program Files (x86)\Steam\steam.exe"),
    row(0, 1300, "G", r"C:\Users\someone\AppData\Local\WhatsApp\WhatsApp.exe"),
    row(0, 1400, "C+G", r"C:\Program Files\PowerToys\PowerToys.exe"),
    row(0, 1500, "C+G", r"C:\Program Files\FanControl\FanControl.exe"),
)
BUSY = table(
    row(0, 1100, "C+G", r"C:\Windows\explorer.exe"),
    row(0, 2000, "C", r"C:\Users\someone\ComfyUI\python_embeded\python.exe", "9000MiB"),
    row(0, 2100, "C", r"C:\Users\someone\Tools\ollama\ollama.exe", "4000MiB"),
    row(1, 2200, "C", r"C:\Tools\llama-server.exe", "2000MiB"),
)
LINUX = table(
    row(0, 3000, "C", "/usr/bin/python3.12", "500MiB"),
)


def check(cfg=None, gpu=0, smi_table=DESKTOP, port=False, ps=None, smi="/usr/bin/nvidia-smi"):
    with mock.patch.object(preflight, "find_nvidia_smi", return_value=smi):
        return preflight.check(cfg or Config(), gpu, run=lambda args, **kw: smi_table,
                               port_open=lambda host, p: port, fetch_ps=lambda url: ps)


class ParseTest(unittest.TestCase):
    def test_both_table_shapes(self):
        rows = preflight.parse_processes(BUSY)
        self.assertEqual([(r["gpu"], r["pid"], r["type"]) for r in rows],
                         [(0, 1100, "C+G"), (0, 2000, "C"), (0, 2100, "C"), (1, 2200, "C")])
        self.assertEqual(rows[1]["memory"], "9000MiB")
        self.assertTrue(rows[1]["name"].endswith("python.exe"))
        self.assertEqual(preflight.parse_processes(LINUX)[0]["name"], "/usr/bin/python3.12")

    def test_garbage_and_empty_output(self):
        self.assertEqual(preflight.parse_processes(""), [])
        self.assertEqual(preflight.parse_processes(None), [])
        self.assertEqual(preflight.parse_processes("No running processes found\n"), [])


class GpuBusyTest(unittest.TestCase):
    def test_desktop_programs_are_not_a_finding(self):
        self.assertEqual(check(smi_table=DESKTOP), [])

    def test_compute_workloads_are_a_finding_with_known_names_only(self):
        rows = check(smi_table=BUSY)
        self.assertEqual([r["kind"] for r in rows], ["gpu_busy"])
        text = rows[0]["text"]
        self.assertIn("2 ", text)
        self.assertIn("ollama", text)
        self.assertIn("python", text)
        for private in ("ComfyUI", "someone", "C:\\", "explorer"):
            self.assertNotIn(private, text)
            self.assertNotIn(private, rows[0]["note"])
        self.assertIn("ComfyUI", rows[0]["detail"])  # console only
        self.assertIn("9000MiB", rows[0]["detail"])
        self.assertNotIn("llama-server", text)  # that one is on GPU 1

    def test_only_the_benchmark_gpu_counts(self):
        rows = check(smi_table=BUSY, gpu=1)
        self.assertEqual(len(rows), 1)
        self.assertIn("llama-server", rows[0]["text"])

    def test_known_engine_counts_even_as_graphics_context(self):
        shown = table(row(0, 5000, "C+G", r"C:\Program Files\LM Studio\LM Studio.exe", "1000MiB"),
                      row(0, 5100, "G", r"C:\Windows\explorer.exe"))
        rows = check(smi_table=shown)
        self.assertEqual(len(rows), 1)
        self.assertIn("lm studio", rows[0]["text"])

    def test_names_in_the_text_are_normalised_and_unique(self):
        shown = table(row(0, 1, "C", "/usr/bin/python3", "1MiB"),
                      row(0, 2, "C", "/usr/bin/python3", "1MiB"))
        text = check(smi_table=shown)[0]["text"]
        self.assertEqual(text.count("python"), 1)
        self.assertIn("2 ", text)

    def test_own_process_is_not_counted(self):
        shown = table(row(0, os.getpid(), "C", "/usr/bin/python3", "1MiB"))
        self.assertEqual(check(smi_table=shown), [])

    def test_unknown_compute_process_counts_without_a_name(self):
        shown = table(row(0, 7, "C", r"C:\Users\someone\secret-tool.exe", "100MiB"))
        rows = check(smi_table=shown)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("secret-tool", rows[0]["text"])
        self.assertIn("secret-tool", rows[0]["detail"])

    def test_without_nvidia_smi_or_on_failure_nothing_is_reported(self):
        self.assertEqual(check(smi_table=BUSY, smi=None), [])
        self.assertEqual(check(smi_table=None), [])


class ForeignOllamaTest(unittest.TestCase):
    def test_listening_on_default_port(self):
        rows = check(port=True, ps={"models": [{"name": "x"}]})
        self.assertEqual([r["kind"] for r in rows], ["foreign_ollama"])
        self.assertIn("11434", rows[0]["text"])
        self.assertIn("1 model", rows[0]["text"])

    def test_given_url_means_it_is_wanted(self):
        self.assertEqual(check(Config(ollama_url="http://127.0.0.1:11434"), port=True), [])

    def test_only_when_ollama_is_selected(self):
        self.assertEqual(check(Config(engines=["llamacpp"]), port=True), [])

    def test_ps_failure_still_warns(self):
        self.assertEqual(len(check(port=True, ps=None)), 1)


if __name__ == "__main__":
    unittest.main()
