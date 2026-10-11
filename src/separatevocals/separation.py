"""Vocal separation using Demucs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def separate(song: Path, outdir: Path, model: str) -> Path:
    """Separate vocals from a song using Demucs.

    Reuses an existing `vocals.wav` if present so the pipeline is
    cheap to re-run with different pitch parameters.
    """
    vocals = outdir / model / song.stem / "vocals.wav"
    if vocals.exists():
        print(f"Reusing {vocals}")
        return vocals
    subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            model,
            "-o",
            str(outdir),
            str(song),
        ],
        check=True,
    )
    return vocals
