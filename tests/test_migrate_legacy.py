import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("migrate_script", ROOT / "scripts" / "migrate_legacy.py")
migrate = importlib.util.module_from_spec(_spec)
sys.modules["migrate_script"] = migrate
_spec.loader.exec_module(migrate)

A = ("# 2026-10-26: PC RTX 3090 24 GB\n\nSee the [README](../README.md).\n\n"
     "## Tests\n\n![x](../charts/a.svg)\n")
B = ("# 2026-10-28: PC RTX 3090 24 GB\n\nSame as the [run from 2026-10-26](2026-10-26-pc-rtx-3090-24gb.md) and "
     "[its logic part](2026-10-26-pc-rtx-3090-24gb.md#logic). "
     "[Site](https://example.com/a.md) [Anchor](#tests)\n")


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class MigrateLegacyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "runs").mkdir()
        (self.root / "runs" / "2026-10-26-pc-rtx-3090-24gb.md").write_text(A, encoding="utf-8")
        (self.root / "runs" / "2026-10-28-pc-rtx-3090-24gb.md").write_text(B, encoding="utf-8")
        git(self.root, "init", "-q")
        git(self.root, "add", ".")
        git(self.root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")

    def test_moves_and_rewrites(self):
        moved = migrate.migrate(self.root)
        self.assertEqual(len(moved), 2)
        runs = self.root / "runs"
        self.assertFalse(list(runs.glob("*.md")))
        a = (runs / "2026-10-26-pc-rtx-3090-24gb" / "README.md").read_text(encoding="utf-8")
        b = (runs / "2026-10-28-pc-rtx-3090-24gb" / "README.md").read_text(encoding="utf-8")
        self.assertIn("(../../README.md)", a)
        self.assertIn("(../../charts/a.svg)", a)
        self.assertIn("(../2026-10-26-pc-rtx-3090-24gb/README.md)", b)
        self.assertIn("(../2026-10-26-pc-rtx-3090-24gb/README.md#logic)", b)
        self.assertIn("(https://example.com/a.md)", b)
        self.assertIn("(#tests)", b)

    def test_results_json(self):
        migrate.migrate(self.root)
        data = json.loads((self.root / "runs" / "2026-10-26-pc-rtx-3090-24gb" / "results.json").read_text())
        self.assertEqual(data, {"schema_version": 1, "legacy": True,
                                "device": "PC RTX 3090 24 GB", "date": "2026-10-26"})

    def test_git_sees_renames(self):
        migrate.migrate(self.root)
        out = subprocess.run(["git", "status", "--short"], cwd=self.root, check=True,
                             capture_output=True, text=True).stdout
        self.assertEqual(len(re.findall(r"^R", out, re.M)), 2, out)

    def test_second_run_does_nothing(self):
        migrate.migrate(self.root)
        self.assertEqual(migrate.migrate(self.root), [])

    def test_works_without_git(self):
        plain = Path(self._tmp.name) / "plain"
        (plain / "runs").mkdir(parents=True)
        (plain / "runs" / "2026-10-26-x.md").write_text(A, encoding="utf-8")
        migrate.migrate(plain)
        self.assertTrue((plain / "runs" / "2026-10-26-x" / "README.md").is_file())

    def test_bad_first_line_is_skipped_not_moved(self):
        (self.root / "runs" / "notes.md").write_text("no title\n", encoding="utf-8")
        migrate.migrate(self.root)
        self.assertTrue((self.root / "runs" / "notes.md").is_file())


class RepoRunsTest(unittest.TestCase):
    """Every relative link in runs/*/README.md points at a file that exists."""

    def test_links(self):
        link = re.compile(r"\]\(([^)\s]+)\)")
        bad = []
        for md in sorted((ROOT / "runs").glob("*/README.md")):
            for target in link.findall(md.read_text(encoding="utf-8")):
                if re.match(r"^([a-z]+:|#)", target):
                    continue
                path = target.split("#", 1)[0]
                if path and not (md.parent / path).exists():
                    bad.append(f"{md.parent.name}: {target}")
        self.assertEqual(bad, [])

    def test_no_report_left_at_old_path(self):
        self.assertEqual(list((ROOT / "runs").glob("*.md")), [])


if __name__ == "__main__":
    unittest.main()
