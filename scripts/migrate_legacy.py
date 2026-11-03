"""One-off move of the pre-harness reports into run folders.

    runs/<date>-<device>.md  ->  runs/<date>-<device>/README.md
                                 runs/<date>-<device>/results.json  (legacy marker)

Relative links are rewritten for the new depth. Nothing stays at the old path.
Uses `git mv` inside a git repository, a plain move otherwise. Running it again
finds no old files and does nothing.

    python3 scripts/migrate_legacy.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLE = re.compile(r"^# (\d{4}-\d{2}-\d{2}): (.+)$")
LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def rewrite_links(text: str) -> str:
    def fix(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^([A-Za-z][A-Za-z0-9+.-]*:|#|/)", target):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        if "/" not in path and path.endswith(".md"):
            new = f"../{path[:-3]}/README.md"  # sibling report
        else:
            new = f"../{path}"
        return f"{m.group(1)}{new}{sep}{anchor}{m.group(3)}"
    return LINK.sub(fix, text)


def _in_git(root: Path) -> bool:
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                           capture_output=True, text=True)
    except OSError:
        return False
    return r.returncode == 0 and Path(r.stdout.strip()).resolve() == root.resolve()


def migrate(root: Path = ROOT) -> list[Path]:
    """Move every runs/*.md with a report title. Returns the new folders."""
    use_git = _in_git(root)
    moved = []
    for old in sorted((root / "runs").glob("*.md")):
        text = old.read_text(encoding="utf-8")
        m = TITLE.match(text.splitlines()[0] if text else "")
        if not m:
            continue
        folder = old.with_suffix("")
        folder.mkdir()
        new = folder / "README.md"
        if use_git:
            subprocess.run(["git", "mv", str(old), str(new)], cwd=root, check=True)
        else:
            shutil.move(str(old), str(new))
        new.write_text(rewrite_links(text), encoding="utf-8")
        data = {"schema_version": 1, "legacy": True, "device": m.group(2).strip(), "date": m.group(1)}
        (folder / "results.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        moved.append(folder)
    return moved


if __name__ == "__main__":
    print(f"moved {len(migrate())} reports")
