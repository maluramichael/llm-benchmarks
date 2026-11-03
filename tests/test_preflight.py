import os
import unittest
from unittest import mock

from bench import preflight
from bench.config import Config

UUIDS = "0, GPU-aaaa\n1, GPU-bbbb\n"


def runner(apps="", uuids=UUIDS):
    def run(args, **_kwargs):
        if any("query-compute-apps" in a for a in args):
            return apps
        if any("query-gpu" in a for a in args):
            return uuids
        return None
    return run


def check(cfg=None, gpu=0, apps="", port=False, uuids=UUIDS, ps=None, smi="/usr/bin/nvidia-smi"):
    with mock.patch.object(preflight, "find_nvidia_smi", return_value=smi):
        return preflight.check(cfg or Config(), gpu, run=runner(apps, uuids),
                               port_open=lambda host, p: port,
                               fetch_ps=lambda url: ps)


class GpuBusyTest(unittest.TestCase):
    def test_clean_machine_has_no_warnings(self):
        self.assertEqual(check(), [])

    def test_other_process_on_the_benchmark_gpu(self):
        rows = check(apps="GPU-aaaa, 999, ComfyUI.exe, 4000\n")
        self.assertEqual([r["kind"] for r in rows], ["gpu_busy"])
        self.assertIn("ComfyUI.exe", rows[0]["text"])  # console only
        self.assertNotIn("ComfyUI", rows[0]["note"])  # what gets stored has no names
        self.assertIn("GPU 0", rows[0]["note"])

    def test_process_on_another_gpu_is_ignored(self):
        self.assertEqual(check(apps="GPU-bbbb, 999, ComfyUI.exe, 4000\n", gpu=0), [])
        self.assertEqual(len(check(apps="GPU-bbbb, 999, ComfyUI.exe, 4000\n", gpu=1)), 1)

    def test_own_process_is_not_counted(self):
        self.assertEqual(check(apps=f"GPU-aaaa, {os.getpid()}, python.exe, 10\n"), [])

    def test_without_nvidia_smi_or_on_failure_nothing_is_reported(self):
        self.assertEqual(check(apps="GPU-aaaa, 9, x.exe, 1\n", smi=None), [])
        self.assertEqual(check(apps=None), [])

    def test_unknown_uuid_mapping_counts_all_processes(self):
        rows = check(apps="GPU-zzzz, 9, x.exe, 1\n", uuids=None)
        self.assertEqual(len(rows), 1)


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
