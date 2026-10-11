"""Tests for the demucs wrapper in src/separatevocals/separation.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from separatevocals import separation  # noqa: E402


def test_separate_reuses_existing_vocals(tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"")
    outdir = tmp_path / "out"
    model = "htdemucs_ft"
    vocals_path = outdir / model / "song" / "vocals.wav"
    vocals_path.parent.mkdir(parents=True)
    vocals_path.write_bytes(b"RIFF....")
    result = separation.separate(song, outdir, model)
    assert result == vocals_path


def test_separate_runs_demucs_when_missing(tmp_path, monkeypatch):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"")
    outdir = tmp_path / "out"
    model = "htdemucs_ft"
    expected = outdir / model / "song" / "vocals.wav"
    expected.parent.mkdir(parents=True)

    calls = {}

    def fake_run(cmd, check):
        calls["cmd"] = cmd
        # Simulate demucs producing the file
        expected.write_bytes(b"RIFF....")
        return None

    monkeypatch.setattr(separation.subprocess, "run", fake_run)
    result = separation.separate(song, outdir, model)
    assert result == expected
    assert "--two-stems=vocals" in calls["cmd"]
    assert str(song) in calls["cmd"]
