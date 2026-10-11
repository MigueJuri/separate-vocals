# Changelog

## 0.1.1 (2026-10-11)

- Fix `track_pitches` dropping all tracks when the first candidate has
  `time=0`. The per-frame duration is now estimated from the median
  positive frame-to-frame delta instead of `candidates[0].time`.
- Add regression tests for the new `_estimate_dt` helper and the
  end-to-end `track_pitches` path with `time=0` on the first frame.

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
