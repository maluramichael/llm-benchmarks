import unittest

from bench.checkers import CHECKERS
from bench.suites.smoke import CATEGORY_COUNTS, load_tasks

SAMPLING = {
    "math": (0.0, None), "code": (0.2, None), "json": (0.0, None), "tool_calls": (0.0, None),
    "long_context": (0.0, None), "creative": (0.8, 0.95), "code_explain": (0.3, None)}


class TaskFilesTest(unittest.TestCase):
    def test_task_files_valid(self):
        tasks = load_tasks(None)
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        counts = {}
        for t in tasks:
            counts[t["category"]] = counts.get(t["category"], 0) + 1
        self.assertEqual(counts, {"math": 8, "code": 6, "json": 5, "tool_calls": 5,
                                  "long_context": 3, "creative": 2, "code_explain": 2})
        self.assertEqual(counts, CATEGORY_COUNTS)

    def test_task_shape_and_sampling(self):
        for t in load_tasks(None):
            self.assertTrue(t["messages"], t["id"])
            for m in t["messages"]:
                self.assertIn(m["role"], ("system", "user", "assistant"))
                self.assertTrue(m["content"].strip())
            self.assertIn(t["check"]["type"], set(CHECKERS) | {"none"}, t["id"])
            temperature, top_p = SAMPLING[t["category"]]
            self.assertEqual(t["sampling"]["temperature"], temperature, t["id"])
            if top_p is not None:
                self.assertEqual(t["sampling"]["top_p"], top_p, t["id"])
            self.assertGreater(t["sampling"]["max_tokens"], 0)

    def test_task_text_is_plain(self):
        import json
        for t in load_tasks(None):
            text = json.dumps(t, ensure_ascii=False)
            self.assertNotIn("—", text, t["id"])
            self.assertTrue(text.isascii(), t["id"])

    def test_check_parameters_present(self):
        for t in load_tasks(None):
            spec = t["check"]
            kind = spec["type"]
            if kind in ("number", "needle"):
                self.assertTrue(spec["answer"], t["id"])
            if kind == "code":
                self.assertIn("assert", spec["tests"])
            if kind == "tool_call":
                self.assertTrue(t.get("tools"), t["id"])
                self.assertTrue(spec.get("expect_none") or spec["name"], t["id"])
            if kind == "json":
                self.assertTrue(t["json_schema"] and spec["expect"], t["id"])
            if t["category"] == "long_context":
                self.assertIn("{document}", t["messages"][-1]["content"])
                self.assertTrue(0 < t["depth"] < 1 and t["needle"])

    def test_pattern_filter(self):
        self.assertEqual({t["category"] for t in load_tasks("math-*")}, {"math"})
        self.assertEqual(len(load_tasks("code,json-1")), 7)
        self.assertEqual(len(load_tasks("tool_calls")), 5)
        self.assertEqual(load_tasks("nothing"), [])


if __name__ == "__main__":
    unittest.main()
