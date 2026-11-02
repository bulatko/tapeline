"""A red gate stops the run; the manifest is written in UTF-8 regardless of the locale."""
import json
import tempfile
from pathlib import Path

import pytest

from tapeline import Config, Pipeline
from tapeline.runner import RedGate


def _pipeline(tmp):
    return Pipeline(Config(frames="x", poses="y", work=tmp))


def test_red_gate_raises_and_is_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        p = _pipeline(tmp)
        with pytest.raises(RedGate):
            p._gate(dict(name="poses", ok=False, components=2))
        m = json.loads(Path(p.cfg.paths()["manifest"]).read_text(encoding="utf-8"))
        assert m["gates"]["poses"]["ok"] is False


def test_green_gate_passes():
    with tempfile.TemporaryDirectory() as tmp:
        p = _pipeline(tmp)
        p._gate(dict(name="cloud", ok=True))
        assert p.manifest["gates"]["cloud"]["ok"]


def test_manifest_written_utf8_non_ascii():
    with tempfile.TemporaryDirectory() as tmp:
        p = _pipeline(tmp)
        p.manifest["stages"]["x"] = {"note": "red gate - na\u00efve caf\u00e9"}
        p._save()
        raw = Path(p.cfg.paths()["manifest"]).read_bytes()
        assert "na\u00efve caf\u00e9".encode("utf-8") in raw
