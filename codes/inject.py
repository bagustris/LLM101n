#!/usr/bin/env python3
"""
inject.py — sync code blocks from codes/chNN/main.py back into chNN.md

This is the reverse of extract.py. It reads named code blocks from a Python
file and patches the corresponding fenced code blocks in the markdown file.

MARKER FORMAT
-------------
In codes/chNN/main.py, wrap each named block with:

    # === block: <name> ===
    ...code...
    # === /block: <name> ===

In chNN.md, wrap the matching fenced code block with HTML comments:

    <!-- block: <name> -->
    ```python
    ...code...
    ```
    <!-- /block: <name> -->

When you run inject.py, the code inside the markdown fence is replaced with
the current content from main.py. Prose between blocks is never touched.

USAGE
-----
    # Sync all chapters
    python inject.py

    # Sync specific chapters
    python inject.py ch01 ch04

    # Dry run (show diffs without writing)
    python inject.py --dry-run

    # Show which chapters have markers
    python inject.py --status
"""

import re
import sys
import difflib
from pathlib import Path

ROOT = Path(__file__).parent.parent  # repo root (parent of codes/)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_RE_PY = re.compile(
    r"# === block: (?P<name>\w+) ===\n"
    r"(?P<code>.*?)"
    r"# === /block: (?P<close>\w+) ===[ \t]*\n",
    re.DOTALL,
)

BLOCK_RE_MD = re.compile(
    r"<!-- block: (?P<name>\w+) -->\n"
    r"```(?P<lang>\w*)\n"
    r"(?P<code>.*?)"
    r"```\n"
    r"<!-- /block: (?P<close>\w+) -->",
    re.DOTALL,
)


def extract_blocks_from_py(py_path: Path) -> dict[str, str]:
    """Return {block_name: code_str} for all marked blocks in a Python file."""
    text = py_path.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for m in BLOCK_RE_PY.finditer(text):
        name, close = m.group("name"), m.group("close")
        if name != close:
            print(f"  WARNING: mismatched block markers: {name!r} vs {close!r}")
            continue
        # Strip trailing newline so the fence looks clean
        blocks[name] = m.group("code").rstrip("\n")
    return blocks


def inject_blocks_into_md(
    md_path: Path,
    blocks: dict[str, str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Replace code inside each marked fence in the markdown file.

    Returns (n_updated, n_skipped) where skipped means the marker was found
    in the markdown but the block name doesn't exist in `blocks`.
    """
    original = md_path.read_text(encoding="utf-8")
    updated = 0
    skipped = 0

    def replacer(m: re.Match) -> str:
        nonlocal updated, skipped
        name, close = m.group("name"), m.group("close")
        if name != close:
            print(f"  WARNING: mismatched md markers: {name!r} vs {close!r}")
            return m.group(0)
        if name not in blocks:
            print(f"  WARNING: marker '{name}' in md but not in main.py — skipping")
            skipped += 1
            return m.group(0)
        lang = m.group("lang") or "python"
        new_code = blocks[name]
        updated += 1
        return (
            f"<!-- block: {name} -->\n"
            f"```{lang}\n"
            f"{new_code}\n"
            f"```\n"
            f"<!-- /block: {name} -->"
        )

    result = BLOCK_RE_MD.sub(replacer, original)

    if result == original:
        return 0, skipped  # nothing changed

    if dry_run:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            result.splitlines(keepends=True),
            fromfile=str(md_path),
            tofile=str(md_path) + " (injected)",
            n=2,
        )
        sys.stdout.writelines(diff)
    else:
        md_path.write_text(result, encoding="utf-8")

    return updated, skipped


def sync_chapter(
    ch_name: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Sync one chapter. Returns True if any changes were made (or would be)."""
    py_path = ROOT / "codes" / ch_name / "main.py"
    md_path = ROOT / f"{ch_name}.md"

    if not py_path.exists():
        if verbose:
            print(f"  {ch_name}: codes/{ch_name}/main.py not found — skip")
        return False
    if not md_path.exists():
        if verbose:
            print(f"  {ch_name}: {ch_name}.md not found — skip")
        return False

    blocks = extract_blocks_from_py(py_path)
    if not blocks:
        if verbose:
            print(f"  {ch_name}: no block markers in main.py")
        return False

    updated, skipped = inject_blocks_into_md(md_path, blocks, dry_run=dry_run)

    if verbose:
        action = "would update" if dry_run else "updated"
        parts = []
        if updated:
            parts.append(f"{updated} block(s) {action}")
        if skipped:
            parts.append(f"{skipped} skipped")
        if not updated and not skipped:
            parts.append("already up to date")
        print(f"  {ch_name}: {', '.join(parts)}")

    return updated > 0


def show_status() -> None:
    """Print which chapters have markers in main.py and/or the markdown."""
    print(f"{'Chapter':<10} {'main.py blocks':<20} {'md fences':<20}")
    print("-" * 50)
    for i in range(1, 18):
        ch = f"ch{i:02d}"
        py_path = ROOT / "codes" / ch / "main.py"
        md_path = ROOT / f"{ch}.md"

        py_blocks: list[str] = []
        md_blocks: list[str] = []

        if py_path.exists():
            text = py_path.read_text(encoding="utf-8")
            py_blocks = [m.group("name") for m in BLOCK_RE_PY.finditer(text)]
        if md_path.exists():
            text = md_path.read_text(encoding="utf-8")
            md_blocks = [m.group("name") for m in BLOCK_RE_MD.finditer(text)]

        py_str = ", ".join(py_blocks) if py_blocks else "(none)"
        md_str = ", ".join(md_blocks) if md_blocks else "(none)"
        print(f"  {ch:<8} {py_str:<20} {md_str:<20}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--status" in args:
        show_status()
        return

    dry_run = "--dry-run" in args
    chapters = [a for a in args if not a.startswith("--")]

    if not chapters:
        chapters = [f"ch{i:02d}" for i in range(1, 18)]

    if dry_run:
        print("Dry run — no files will be modified.\n")

    changed = 0
    for ch in chapters:
        if sync_chapter(ch, dry_run=dry_run):
            changed += 1

    print(f"\nDone. {changed}/{len(chapters)} chapter(s) {'would be' if dry_run else ''} updated.")


if __name__ == "__main__":
    main()
