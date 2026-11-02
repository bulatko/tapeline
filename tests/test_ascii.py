"""Source and documentation files are plain ASCII, apart from a few typographic symbols used in the docs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".md", ".toml", ".cff"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", "runs"}
ALLOWED = set("\u2265\u00b7\u00b5\u2013\u2500\u25ba")  # greater-or-equal, middle dot, micro, en dash, box line, pointer


def non_ascii(text):
    """-> [(line_no, char)] for characters outside ASCII and ALLOWED."""
    return [(i, c) for i, line in enumerate(text.splitlines(), 1) for c in line if ord(c) > 127 and c not in ALLOWED]


def test_files_are_ascii():
    bad = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and p.suffix in SUFFIXES and not SKIP_DIRS.intersection(rel.parts):
            bad += [f"{rel}:{i}: U+{ord(c):04X}" for i, c in non_ascii(p.read_text(encoding="utf-8"))]
    assert not bad, "non-ASCII characters:\n" + "\n".join(bad[:50])


def test_checker_flags_non_ascii_and_accepts_allowed():
    assert non_ascii("x \u2192 y") == [(1, "\u2192")]
    assert non_ascii("a \u2265 b, 2\u00b7voxel, \u00b5, 1\u20133") == []
