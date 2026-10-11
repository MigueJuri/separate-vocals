"""Lightweight tests for the pure-numpy helpers in vocal_notes.

We deliberately avoid exercising the full pipeline because it requires
GPU/CPU-bound Demucs, optional pyannote.audio, and audio I/O. These
tests cover the deterministic building blocks a reviewer can run
quickly in CI.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "vocal_notes.py"


def _load_vocal_notes():
    spec = importlib.util.spec_from_file_location("vocal_notes", SCRIPT)
    assert spec and spec.loader, f"Could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["vocal_notes"] = module
    spec.loader.exec_module(module)
    return module


vocal_notes = _load_vocal_notes()


def test_cents_diff_identical_is_zero():
    assert vocal_notes._cents_diff(440.0, 440.0) == pytest.approx(0.0)


def test_cents_diff_one_semitone():
    # 1 semitone = 100 cents
    a = 440.0
    b = 440.0 * (2 ** (1 / 12))
    assert vocal_notes._cents_diff(a, b) == pytest.approx(100.0, rel=1e-3)


def test_cents_diff_invalid_inputs_are_huge():
    assert vocal_notes._cents_diff(0.0, 440.0) > 1e6
    assert vocal_notes._cents_diff(440.0, -1.0) > 1e6


def test_estimate_dt_handles_first_frame_zero_time():
    # The first candidate has time=0 (which previously broke the
    # duration estimate). _estimate_dt should ignore that and recover
    # the real frame step from the rest of the candidates.
    dt = 256 / 22050
    candidates = [
        vocal_notes.PitchCandidate(
            frame_idx=0, time=0.0, f0_hz=220.0, midi=57.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        ),
        vocal_notes.PitchCandidate(
            frame_idx=1, time=dt, f0_hz=220.0, midi=57.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        ),
        vocal_notes.PitchCandidate(
            frame_idx=2, time=2 * dt, f0_hz=220.0, midi=57.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        ),
    ]
    assert vocal_notes._estimate_dt(candidates) == pytest.approx(dt, rel=1e-6)


def test_track_pitches_keeps_track_when_first_time_is_zero():
    # Regression: with the first candidate's time=0, the previous
    # implementation computed dt=0 and dropped the only track.
    dt = 256 / 22050
    candidates = [
        vocal_notes.PitchCandidate(
            frame_idx=t, time=t * dt, f0_hz=220.0, midi=57.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        )
        for t in range(20)
    ]
    tracks = vocal_notes.track_pitches(candidates, max_voices=2, min_dur=0.04)
    assert len(tracks) == 1
    assert tracks[0].start_frame == 0
    assert tracks[0].end_frame == 19
    # duration must be a real number of seconds, not zero
    assert (tracks[0].end_frame - tracks[0].start_frame + 1) * dt > 0.04


def test_pitch_track_to_note_rounding():
    tr = vocal_notes.PitchTrack(track_id=0, start_frame=10, end_frame=19)
    tr.frames = list(range(10, 20))
    # Constant A4 = 440 Hz
    tr.f0_hz = np.full(10, 440.0)
    tr.magnitude = np.full(10, 0.5)
    tr.confidence = np.full(10, 0.9)
    dt = 256 / 22050
    s, e, midi, amp = tr.to_note(dt)
    assert midi == 69  # A4
    assert amp == pytest.approx(0.5)
    assert s == pytest.approx(10 * dt)
    assert e == pytest.approx(20 * dt)


def test_track_pitches_merges_close_pitch_across_short_gap():
    # Two candidates 350 cents apart, in adjacent frames, then a 5-frame
    # gap, then a third candidate near the second. The merger should
    # collapse them into a single track.
    dt = 256 / 22050
    f0_a = 220.0  # A3
    f0_b = 220.0 * (2 ** (350 / 1200))  # ~350 cents above
    candidates = []
    for t in range(0, 5):
        candidates.append(vocal_notes.PitchCandidate(
            frame_idx=t, time=t * dt, f0_hz=f0_a, midi=57.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        ))
    # gap of 5 frames
    for t in range(10, 15):
        candidates.append(vocal_notes.PitchCandidate(
            frame_idx=t, time=t * dt, f0_hz=f0_b, midi=int(57 + 3.5),
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        ))
    tracks = vocal_notes.track_pitches(
        candidates,
        max_voices=2,
        max_jump_cents=400.0,
        merge_gap_cents=400.0,
        merge_gap_frames=8,
    )
    # The two close-in-pitch fragments should merge into one track.
    assert len(tracks) == 1
    assert tracks[0].start_frame == 0
    assert tracks[0].end_frame == 14


def test_track_pitches_drops_short_tracks():
    dt = 256 / 22050
    cands = [
        vocal_notes.PitchCandidate(
            frame_idx=0, time=0.0, f0_hz=440.0, midi=69.0,
            magnitude=0.5, confidence=0.9, harmonic_strength=0.5,
        )
    ]
    tracks = vocal_notes.track_pitches(cands, max_voices=2, min_dur=0.5)
    assert tracks == []


def test_assign_tracks_by_pitch_prefers_continuity():
    tr_a = vocal_notes.PitchTrack(track_id=0, start_frame=0, end_frame=5)
    tr_a.frames = list(range(6))
    tr_a.f0_hz = np.full(6, 440.0)
    tr_b = vocal_notes.PitchTrack(track_id=1, start_frame=6, end_frame=10)
    tr_b.frames = list(range(6, 11))
    tr_b.f0_hz = np.full(5, 440.0)
    labels = vocal_notes._assign_tracks_by_pitch([tr_a, tr_b], n_voices=2)
    assert labels[0] == labels[1]


def test_refine_f0_snap_to_harmonic():
    # Build a magnitude that has a clear comb at 440 Hz, slightly off
    # nominal f0. refine_f0 should snap the f0 to the strongest comb.
    sr = 22050
    n_fft = 2048
    bin_hz = sr / n_fft
    n_frames = 1
    n_bins = n_fft // 2 + 1
    mag = np.zeros((n_bins, n_frames), dtype=np.float32)
    f0_nominal = 440.0
    H = 10
    # Place energy slightly off from f0_nominal so the snap moves it.
    for k in range(1, H + 1):
        true_hz = 442.0 * k
        b = int(round(true_hz / bin_hz))
        if 0 <= b < n_bins:
            mag[b, 0] = 1.0
    f0 = np.array([[f0_nominal], [np.nan]], dtype=np.float32)
    refined = vocal_notes.refine_f0(mag, f0, sr, n_fft, H=H, cents=80)
    assert abs(refined[0, 0] - 442.0) < 1.0


def test_f0_tracks_from_contours_pads_edges():
    sr, hop = 22050, 256
    n_frames = 50
    tr = vocal_notes.PitchTrack(track_id=0, start_frame=10, end_frame=20)
    tr.frames = list(range(10, 21))
    tr.f0_hz = np.full(11, 330.0)
    tr.magnitude = np.full(11, 0.5)
    tr.confidence = np.full(11, 0.9)
    f0 = vocal_notes.f0_tracks_from_contours(
        [tr], [0], n_voices=1, n_frames=n_frames, sr=sr, hop=hop, pad=3,
    )
    # After padding, the contour extends a few frames before and after.
    active = np.where(~np.isnan(f0[0]))[0]
    assert active[0] == 7
    assert active[-1] == 23


def test_harmonic_masks_shape_and_sum_to_one():
    sr, n_fft = 22050, 1024
    n_frames = 32
    n_bins = n_fft // 2 + 1
    mag = np.ones((n_bins, n_frames), dtype=np.float32)
    f0 = np.full((2, n_frames), np.nan, dtype=np.float32)
    f0[0, :] = 220.0
    f0[1, :] = 330.0
    masks = vocal_notes.harmonic_masks(
        f0, mag, sr, n_fft, H=8, hw0=30.0, hwr=0.02,
        smooth_frames=1, min_mask=0.05, shared_min=0.10,
    )
    assert masks.shape == (2, n_bins, n_frames)
    sums = masks.sum(axis=0)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
