# Changelog

## 0.1.0 (2026-10-11)

- Initial public release.
- Demucs-based vocal separation with `htdemucs_ft`.
- Multi-pitch candidate detection with hysteresis and harmonic-strength
  scoring.
- Hungarian-assignment multi-pitch tracker with gap interpolation and
  short-track merging.
- Speaker diarization via pyannote.audio (preferred) and speechbrain
  (token-free fallback).
- Per-voice harmonic masks with shared-harmonic protection.
- MIDI, CSV and per-voice WAV export, plus diagnostic plots.
- CLI entry point `separate-vocals` and Python module
  `separatevocals.vocal_notes`.
