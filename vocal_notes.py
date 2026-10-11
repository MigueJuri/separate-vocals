#!/usr/bin/env python3
"""
song -> vocal stem (Demucs) -> notes per voice -> per-voice .wav

Phase 0/1/2: diagnostics + anti-dropout tracking + multi-F0 Hungarian tracker
             + speaker-aware identity assignment for overlapping voices.

pip install demucs librosa scipy numpy soundfile pretty_midi matplotlib
pip install pyannote.audio             # recommended backend
# or
pip install speechbrain scikit-learn   # fallback backend (no HF token needed)

Usage:
  python vocal_notes.py song.mp3                       # multi-pitch + pyannote
  python vocal_notes.py song.mp3 --diarizer speechbrain
  python vocal_notes.py song.mp3 --diarizer none       # legacy pitch-continuity
  python vocal_notes.py song.mp3 --mode pyin           # solo voice
  python vocal_notes.py song.mp3 --split stereo        # hard-panned L/R

Env (pyannote only):
  HF_TOKEN=huggingface_token
  Accept licenses at:
    https://huggingface.co/pyannote/segmentation-3.0
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/speaker-diarization-community-1
"""
import argparse, csv, json, os, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import librosa, pretty_midi, soundfile as sf
from scipy.signal import medfilt
from scipy.optimize import linear_sum_assignment

matplotlib_backend_set = False

# ---------- 0. Config ----------
@dataclass
class PitchCandidate:
    frame_idx: int
    time: float
    f0_hz: float
    midi: float
    magnitude: float
    confidence: float
    harmonic_strength: float

@dataclass
class PitchTrack:
    track_id: int
    start_frame: int
    end_frame: int
    frames: List[int] = field(default_factory=list)
    f0_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    magnitude: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))

    def mean_f0(self) -> float:
        if len(self.f0_hz) == 0:
            return 0.0
        return float(np.mean(self.f0_hz))

    def mean_confidence(self) -> float:
        if len(self.confidence) == 0:
            return 0.0
        return float(np.mean(self.confidence))

    def mean_midi(self) -> float:
        if len(self.f0_hz) == 0:
            return 0.0
        return float(np.mean(librosa.hz_to_midi(self.f0_hz)))

    def to_note(self, dt: float) -> Tuple[float, float, int, float]:
        start = self.frames[0] * dt
        end = (self.frames[-1] + 1) * dt
        midi = int(round(self.mean_midi()))
        amp = float(np.mean(self.magnitude))
        return start, end, midi, amp

# ---------- 1. Separation ----------
def separate(song: Path, outdir: Path, model: str) -> Path:
    vocals = outdir / model / song.stem / "vocals.wav"
    if vocals.exists():
        print(f"Reusing {vocals}")
        return vocals
    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                    "-n", model, "-o", str(outdir), str(song)], check=True)
    return vocals

# ---------- 2a. Multi-pitch candidate detection ----------
def detect_pitch_candidates(vocals: Path, fmin: str, fmax: str,
                            max_pitches: int = 4,
                            onset_thresh_db: float = -24.0,
                            maintain_thresh_db: float = -36.0,
                            sr: int = 22050, n_fft: int = 2048, hop: int = 256,
                            H: int = 5) -> Tuple[List[PitchCandidate], np.ndarray, np.ndarray, np.ndarray]:
    """Detect pitch candidates per frame with hysteresis and harmonic strength.

    Returns candidates, STFT magnitude, times, frequencies.
    """
    y, _ = librosa.load(str(vocals), sr=sr, mono=True)
    pitches, mags = librosa.piptrack(
        y=y, sr=sr,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        n_fft=n_fft, hop_length=hop,
    )

    n_frames = pitches.shape[1]
    times = librosa.times_like(pitches, sr=sr, hop_length=hop)
    dt = float(times[1] - times[0]) if n_frames > 1 else (hop / sr)
    ref = float(mags.max()) + 1e-10
    onset_rel = 10 ** (onset_thresh_db / 20.0)
    maintain_rel = 10 ** (maintain_thresh_db / 20.0)
    bin_hz = sr / n_fft
    freqs = np.arange(n_fft // 2 + 1) * bin_hz
    k = np.arange(1, H + 1)

    # Build STFT magnitude for harmonic-strength computation
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    full_mag = np.abs(D)

    # Find local maxima per column and their harmonic strength
    candidates = []
    prev_peaks = []  # for hysteresis: (midi, f0_hz)
    for t in range(n_frames):
        col_p = pitches[:, t]
        col_m = mags[:, t]
        peaks = []
        for f in range(1, len(col_p) - 1):
            p = col_p[f]
            m = col_m[f]
            if p <= 0 or m <= 0:
                continue
            if col_m[f-1] > m or col_m[f+1] > m:
                continue
            # Hysteresis: easier to maintain a peak than to start one
            midi = librosa.hz_to_midi(float(p))
            if np.isnan(midi):
                continue
            midi_round = int(round(midi))
            prev_match = any(abs(midi_round - pm) <= 1 for pm, _ in prev_peaks)
            thresh = maintain_rel if prev_match else onset_rel
            if m < ref * thresh:
                continue

            # Harmonic strength: energy around harmonics in the full spectrogram
            harm_hz = p * k
            harm_bins = np.clip(np.rint(harm_hz / bin_hz).astype(int), 0, full_mag.shape[0] - 1)
            harm_strength = float(full_mag[harm_bins, t].sum()) / max(1e-10, float(full_mag[:, t].sum()))

            conf = (m / ref) * (1 + 5 * harm_strength)
            peaks.append(PitchCandidate(
                frame_idx=t, time=times[t], f0_hz=float(p), midi=midi,
                magnitude=float(m), confidence=float(conf),
                harmonic_strength=float(harm_strength),
            ))
        peaks.sort(key=lambda c: -c.confidence)
        # Also keep peaks that are close to previous peaks even if lower confidence
        kept = peaks[:max_pitches]
        prev_peaks = [(int(round(c.midi)), c.f0_hz) for c in kept]
        candidates.extend(kept)

    return candidates, full_mag, times, freqs

# ---------- 2b. Multi-pitch tracking (Hungarian per frame) ----------
def _cents_diff(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return 1e9
    return 1200.0 * abs(np.log2(f1 / f2))


def _estimate_dt(candidates: List[PitchCandidate], default: float = 0.01) -> float:
    """Robust per-frame duration estimate.

    The previous implementation used ``candidates[0].time`` which is
    ``0.0`` for the first frame and silently zeroed every duration
    check, causing the close-remaining branch to drop the only track.
    Instead, take the median of positive frame-to-frame deltas.
    """
    if not candidates:
        return default
    times = sorted({c.time for c in candidates if c.time > 0.0})
    if len(times) < 2:
        return default
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return default
    return float(np.median(deltas))


def track_pitches(candidates: List[PitchCandidate], max_voices: int = 2,
                  max_jump_cents: float = 350.0, gap_frames: int = 15,
                  min_dur: float = 0.04,
                  merge_gap_cents: float = 250.0,
                  merge_gap_frames: int = 12) -> List[PitchTrack]:
    """Track pitch contours using Hungarian assignment per frame.

    Maintains up to max_voices simultaneous tracks. Resolves crossing
    trajectories globally within each frame. Post-process merges short gaps
    and connects nearby tracks to reduce fragmentation.
    """
    if not candidates:
        return []

    by_frame = {}
    for c in candidates:
        by_frame.setdefault(c.frame_idx, []).append(c)
    frames = sorted(by_frame.keys())
    if not frames:
        return []

    dt = _estimate_dt(candidates)

    tracks = []
    next_id = 0
    active = []  # list of PitchTrack

    for t in frames:
        cands = by_frame[t]
        n_active = len(active)
        n_cands = len(cands)

        if n_active == 0:
            # Start new tracks for top candidates
            for c in cands[:max_voices]:
                tr = PitchTrack(track_id=next_id, start_frame=t, end_frame=t)
                next_id += 1
                tr.frames = [t]
                tr.f0_hz = np.array([c.f0_hz])
                tr.magnitude = np.array([c.magnitude])
                tr.confidence = np.array([c.confidence])
                active.append(tr)
            continue

        # Build cost matrix between active tracks and current candidates
        cost = np.full((n_active, n_cands), 1e9, dtype=np.float64)
        for i, tr in enumerate(active):
            last_f0 = tr.f0_hz[-1]
            for j, c in enumerate(cands):
                d = _cents_diff(last_f0, c.f0_hz)
                if d > max_jump_cents:
                    continue
                # Prefer candidates with higher confidence, penalize large jumps
                cost[i, j] = d + 50.0 * (1.0 - min(1.0, c.confidence))

        # Hungarian assignment
        rows, cols = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_cands = set()
        for r, c_idx in zip(rows, cols):
            if cost[r, c_idx] >= 1e8:
                continue
            tr = active[r]
            cand = cands[c_idx]
            tr.frames.append(t)
            tr.f0_hz = np.append(tr.f0_hz, cand.f0_hz)
            tr.magnitude = np.append(tr.magnitude, cand.magnitude)
            tr.confidence = np.append(tr.confidence, cand.confidence)
            tr.end_frame = t
            assigned_tracks.add(r)
            assigned_cands.add(c_idx)

        # Terminate unassigned tracks
        new_active = []
        for i, tr in enumerate(active):
            if i in assigned_tracks:
                new_active.append(tr)
            elif t - tr.end_frame <= gap_frames:
                # Gap: keep alive, interpolate
                tr.frames.append(t)
                tr.f0_hz = np.append(tr.f0_hz, tr.f0_hz[-1])
                tr.magnitude = np.append(tr.magnitude, 0.0)
                tr.confidence = np.append(tr.confidence, 0.0)
                tr.end_frame = t
                new_active.append(tr)
            else:
                # Dead track
                if (tr.end_frame - tr.start_frame + 1) * dt >= min_dur:
                    tracks.append(tr)
        active = new_active

        # Start new tracks for unassigned candidates
        for j, c in enumerate(cands):
            if j in assigned_cands:
                continue
            if len(active) >= max_voices:
                break
            tr = PitchTrack(track_id=next_id, start_frame=t, end_frame=t)
            next_id += 1
            tr.frames = [t]
            tr.f0_hz = np.array([c.f0_hz])
            tr.magnitude = np.array([c.magnitude])
            tr.confidence = np.array([c.confidence])
            active.append(tr)

    # Close remaining active tracks
    for tr in active:
        if (tr.end_frame - tr.start_frame + 1) * dt >= min_dur:
            tracks.append(tr)

    # Post-process 1: fill small internal gaps by interpolation
    for tr in tracks:
        gap_idx = np.where(tr.magnitude == 0.0)[0]
        if len(gap_idx) == 0:
            continue
        good = np.where(tr.magnitude > 0.0)[0]
        if len(good) < 2:
            continue
        tr.f0_hz = np.interp(np.arange(len(tr.f0_hz)), good, tr.f0_hz[good])
        tr.confidence = np.interp(np.arange(len(tr.confidence)), good, tr.confidence[good])

    # Post-process 2: merge tracks that are close in pitch and time
    merged = True
    while merged and len(tracks) > 1:
        merged = False
        tracks.sort(key=lambda tr: tr.start_frame)
        for i in range(len(tracks)):
            if merged:
                break
            for j in range(i + 1, len(tracks)):
                ti, tj = tracks[i], tracks[j]
                gap = tj.start_frame - ti.end_frame - 1
                if gap < 0 or gap > merge_gap_frames:
                    continue
                d = _cents_diff(ti.f0_hz[-1], tj.f0_hz[0])
                if d > merge_gap_cents:
                    continue
                # Merge j into i
                fill_frames = list(range(ti.end_frame + 1, tj.start_frame))
                ti.frames.extend(fill_frames)
                ti.frames.extend(tj.frames)
                # Interpolate f0 across the gap
                if fill_frames:
                    start_hz = ti.f0_hz[-1]
                    end_hz = tj.f0_hz[0]
                    inter_f0 = np.linspace(start_hz, end_hz, len(fill_frames) + 2)[1:-1]
                    inter_mag = np.zeros(len(fill_frames))
                    inter_conf = np.zeros(len(fill_frames))
                    ti.f0_hz = np.concatenate([ti.f0_hz, inter_f0, tj.f0_hz])
                    ti.magnitude = np.concatenate([ti.magnitude, inter_mag, tj.magnitude])
                    ti.confidence = np.concatenate([ti.confidence, inter_conf, tj.confidence])
                else:
                    ti.f0_hz = np.concatenate([ti.f0_hz, tj.f0_hz])
                    ti.magnitude = np.concatenate([ti.magnitude, tj.magnitude])
                    ti.confidence = np.concatenate([ti.confidence, tj.confidence])
                ti.end_frame = tj.end_frame
                tracks.pop(j)
                merged = True
                break

    tracks.sort(key=lambda tr: (tr.start_frame, -tr.mean_confidence()))
    return tracks

# ---------- 2c. Speaker diarization and embeddings ----------
def diarize_pyannote(vocals: Path, n_speakers: int, hf_token: str = None):
    """Diarize with pyannote-audio 3.1. Requires a HuggingFace token.

    Loads the audio with soundfile and passes it as a pre-loaded waveform
    to pyannote, avoiding torchcodec (which requires FFmpeg 5 libs not
    available on FFmpeg-7-only systems like Ubuntu 26.04).

    Returns (segments, embeddings_timeline) where embeddings_timeline is a
    list of (center_time, speaker_embedding) tuples, or None if extraction fails.
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            import huggingface_hub
            hf_token = huggingface_hub.get_token()
        except Exception:
            hf_token = None
    if not hf_token:
        raise RuntimeError(
            "pyannote.audio needs a HuggingFace token. Set HF_TOKEN env var, "
            "pass --hf-token, or run `huggingface_hub login`. Also accept "
            "licenses at:\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline, Inference
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    y, sr = sf.read(str(vocals), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(y.T)  # (channels, time)

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    audio_in = {"waveform": waveform, "sample_rate": sr}
    with ProgressHook() as hook:
        output = pipeline(audio_in, min_speakers=n_speakers,
                          max_speakers=n_speakers, hook=hook)
    diarization = getattr(output, "speaker_diarization", output)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        idx = int(speaker.split("_")[-1])
        segments.append((float(turn.start), float(turn.end), idx))

    # Extract speaker embeddings per short window for identity disambiguation
    embeddings_timeline = None
    try:
        from pyannote.audio import Model
        emb_model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            token=hf_token,
        )
        if torch.cuda.is_available():
            emb_model.to(torch.device("cuda"))
        seg_model = Inference(emb_model, window="sliding", duration=1.0, step=0.5)
        emb_iter = seg_model(audio_in)
        embeddings_timeline = []
        for (start, stop), emb in emb_iter:
            embeddings_timeline.append(((start + stop) / 2.0, emb))
    except Exception as e:
        print(f"      could not extract speaker embeddings: {e}")

    return segments, embeddings_timeline


def diarize_speechbrain(vocals: Path, n_speakers: int):
    """Diarize with speechbrain ECAPA-TDNN + spectral clustering.
    Self-contained: no HF token needed, model is downloaded on first use."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from sklearn.cluster import SpectralClustering

    cache = Path.home() / ".cache" / "speechbrain_ecapa"
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache),
    )

    sr = 16000
    win_s, hop_s = 1.5, 0.75
    y, _ = librosa.load(str(vocals), sr=sr, mono=True)
    win = int(win_s * sr)
    hop = int(hop_s * sr)

    embeddings, centers = [], []
    with torch.no_grad():
        for start in range(0, len(y) - win, hop):
            seg = y[start:start + win]
            t = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)
            emb = model.encode_batch(t)
            embeddings.append(emb.squeeze().cpu().numpy())
            centers.append((start + win // 2) / sr)

    X = np.stack(embeddings)
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    S = (X / norm) @ (X / norm).T
    S = (S + 1.0) / 2.0

    labels = SpectralClustering(
        n_clusters=n_speakers, affinity="precomputed", random_state=0
    ).fit_predict(S)

    segments = []
    cur = int(labels[0])
    seg_start = max(0.0, centers[0] - win_s / 2)
    for i in range(1, len(labels)):
        if labels[i] != cur:
            seg_end = centers[i-1] + win_s / 2
            segments.append((seg_start, seg_end, cur))
            cur = int(labels[i])
            seg_start = centers[i] - win_s / 2
    segments.append((seg_start, centers[-1] + win_s / 2, cur))

    embeddings_timeline = list(zip(centers, embeddings))
    return segments, embeddings_timeline


def diarize_speakers(vocals: Path, n_speakers: int, backend: str = "pyannote",
                     hf_token: str = None):
    if backend == "none":
        return None, None
    if backend == "pyannote":
        try:
            return diarize_pyannote(vocals, n_speakers, hf_token)
        except ImportError:
            print("pyannote.audio not installed, falling back to speechbrain")
            return diarize_speechbrain(vocals, n_speakers)
    if backend == "speechbrain":
        return diarize_speechbrain(vocals, n_speakers)
    raise ValueError(f"Unknown diarizer: {backend!r}")

# ---------- 2d. Track identity assignment ----------
def _track_embedding(track: PitchTrack, embeddings_timeline,
                     sr: int, hop: int) -> Optional[np.ndarray]:
    """Average speaker embeddings over the frames of a track."""
    if not embeddings_timeline:
        return None
    dt = hop / sr
    times = [frame * dt for frame in track.frames]
    emb_list = []
    for t in times:
        # find nearest embedding window center
        best = min(embeddings_timeline, key=lambda x: abs(x[0] - t))
        if abs(best[0] - t) < 1.5:
            emb_list.append(best[1])
    if not emb_list:
        return None
    return np.mean(emb_list, axis=0)


def assign_track_identities(tracks: List[PitchTrack], segments,
                            embeddings_timeline, n_voices: int,
                            vocals: Path, sr: int, hop: int) -> List[int]:
    """Assign each track to a voice using speaker overlap + embeddings.

    For overlapping sections, uses speaker embeddings and pitch continuity
    to avoid assigning two simultaneous tracks to the same voice.
    """
    if n_voices == 1 or not tracks:
        return [0] * len(tracks)
    if segments is None:
        return _assign_tracks_by_pitch(tracks, n_voices)

    dt = hop / sr

    # Step 1: speaker overlap score per track
    overlap_scores = []
    for tr in tracks:
        s = tr.start_frame * dt
        e = (tr.end_frame + 1) * dt
        overlaps = [0.0] * n_voices
        for seg_s, seg_e, spk in segments:
            if spk >= n_voices:
                continue
            ov_s = max(s, seg_s)
            ov_e = min(e, seg_e)
            if ov_e > ov_s:
                overlaps[spk] += ov_e - ov_s
        overlap_scores.append(overlaps)

    # Step 2: speaker embedding per track and build voice prototypes from
    # tracks that have >80% overlap with a single speaker.
    track_embs = [_track_embedding(tr, embeddings_timeline, sr, hop) for tr in tracks]
    voice_protos = [[] for _ in range(n_voices)]
    for i, (tr, emb) in enumerate(zip(tracks, track_embs)):
        if emb is None:
            continue
        total = sum(overlap_scores[i])
        if total > 0:
            winner = int(np.argmax(overlap_scores[i]))
            share = overlap_scores[i][winner] / total
            if share >= 0.7:
                voice_protos[winner].append(emb)
    voice_protos = [np.mean(p, axis=0) if p else None for p in voice_protos]

    # Step 3: build cost matrix for assignment. We assign all tracks at once
    # considering both speaker evidence and pitch-continuity / collision constraints.
    labels = [-1] * len(tracks)

    # First assign tracks with strong speaker evidence
    for i, tr in enumerate(tracks):
        total = sum(overlap_scores[i])
        if total > 0:
            winner = int(np.argmax(overlap_scores[i]))
            share = overlap_scores[i][winner] / total
            if share >= 0.7:
                labels[i] = winner

    # Then use embedding similarity for unassigned tracks
    if any(p is not None for p in voice_protos):
        for i, tr in enumerate(tracks):
            if labels[i] != -1:
                continue
            emb = track_embs[i]
            if emb is None:
                continue
            sims = []
            for v in range(n_voices):
                if voice_protos[v] is None:
                    sims.append(-1.0)
                else:
                    sim = np.dot(emb, voice_protos[v]) / (np.linalg.norm(emb) * np.linalg.norm(voice_protos[v]) + 1e-8)
                    sims.append(sim)
            labels[i] = int(np.argmax(sims))

    # Fallback: pitch continuity for remaining tracks, but resolve collisions
    remaining = [i for i, l in enumerate(labels) if l == -1]
    if remaining:
        # Sort by start time
        remaining.sort(key=lambda i: tracks[i].start_frame)
        last_pitch = [None] * n_voices
        for i in remaining:
            tr = tracks[i]
            midi = tr.mean_midi()
            # Find voice whose last pitch is closest and not currently occupied
            best, best_dist = 0, 1e9
            for v in range(n_voices):
                if last_pitch[v] is not None:
                    d = abs(last_pitch[v] - midi)
                    if d < best_dist:
                        best_dist, best = d, v
            labels[i] = best
            last_pitch[best] = midi

    # Step 4: Resolve collisions at overlapping frames. If two tracks with
    # same label overlap in time, reassign the weaker one.
    for _ in range(3):
        changed = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if labels[i] == labels[j]:
                    # Check temporal overlap
                    si, ei = tracks[i].start_frame, tracks[i].end_frame
                    sj, ej = tracks[j].start_frame, tracks[j].end_frame
                    if not (ei < sj or ej < si):
                        # Keep the one with higher speaker evidence / confidence
                        ci = tracks[i].mean_confidence()
                        cj = tracks[j].mean_confidence()
                        # Prefer track whose overlap score is more concentrated
                        fi = max(overlap_scores[i]) if sum(overlap_scores[i]) > 0 else 0
                        fj = max(overlap_scores[j]) if sum(overlap_scores[j]) > 0 else 0
                        weak = j if (fi * ci > fj * cj) else i
                        # Assign weak to the other voice
                        labels[weak] = 1 - labels[weak] if n_voices == 2 else (labels[weak] + 1) % n_voices
                        changed = True
        if not changed:
            break

    # Step 5: remap so voice 0 has higher mean pitch
    means = []
    for v in range(n_voices):
        mids = [tracks[i].mean_midi() for i in range(len(tracks)) if labels[i] == v]
        means.append(float(np.mean(mids)) if mids else 0.0)
    order = np.argsort(means)[::-1]
    remap = {int(old): new for new, old in enumerate(order)}
    return [remap[l] for l in labels]


def _assign_tracks_by_pitch(tracks: List[PitchTrack], n_voices: int) -> List[int]:
    """Fallback pitch-continuity assignment for tracks."""
    labels = []
    last_pitch = [None] * n_voices
    for tr in tracks:
        midi = tr.mean_midi()
        best, best_dist = 0, 1e9
        for v in range(n_voices):
            if last_pitch[v] is not None:
                d = abs(last_pitch[v] - midi)
                if d < best_dist:
                    best_dist, best = d, v
        labels.append(best)
        last_pitch[best] = midi
    means = []
    for v in range(n_voices):
        mids = [tracks[i].mean_midi() for i in range(len(tracks)) if labels[i] == v]
        means.append(float(np.mean(mids)) if mids else 0.0)
    order = np.argsort(means)[::-1]
    remap = {int(old): new for new, old in enumerate(order)}
    return [remap[l] for l in labels]

# ---------- 3. Note and F0 export ----------
def tracks_to_notes(tracks: List[PitchTrack], labels: List[int],
                    dt: float, n_voices: int):
    """Convert tracks to note events and per-voice note lists."""
    notes = []
    for tr, lab in zip(tracks, labels):
        s, e, midi, amp = tr.to_note(dt)
        notes.append((s, e, midi, amp, lab, tr))
    notes.sort(key=lambda x: x[0])
    return notes


def fill_f0_by_diarization(f0: np.ndarray, segments, n_voices: int,
                           sr: int, hop: int,
                           max_fill_gap: int = 30) -> np.ndarray:
    """Forward-fill / interpolate F0 inside speaker diarization segments.

    This keeps a voice from dropping out when pitch detection misses a few
    frames but the speaker is still active.
    """
    if segments is None:
        return f0
    dt = hop / sr
    out = f0.copy()
    for v in range(n_voices):
        # Build active mask from diarization for this voice
        active = np.zeros(f0.shape[1], dtype=bool)
        for seg_s, seg_e, spk in segments:
            if spk != v:
                continue
            sf = max(0, int(seg_s / dt))
            ef = min(f0.shape[1], int(np.ceil(seg_e / dt)))
            active[sf:ef] = True

        f = out[v].copy()
        # Forward fill short NaN gaps inside active regions
        last = None
        gap_start = None
        for t in range(len(f)):
            if active[t] and np.isnan(f[t]):
                if gap_start is None:
                    gap_start = t
            else:
                if gap_start is not None and last is not None:
                    gap_end = t
                    gap_len = gap_end - gap_start
                    if gap_len <= max_fill_gap:
                        for k in range(gap_start, gap_end):
                            f[k] = last
                if not np.isnan(f[t]):
                    last = f[t]
                gap_start = None
        # Fill trailing gap inside active region
        if gap_start is not None and last is not None:
            gap_len = len(f) - gap_start
            if gap_len <= max_fill_gap:
                f[gap_start:] = last
        out[v] = f
    return out


def f0_tracks_from_contours(tracks: List[PitchTrack], labels: List[int],
                            n_voices: int, n_frames: int, sr: int, hop: int,
                            pad: int = 5,
                            max_interp_gap: int = 20) -> np.ndarray:
    """Build per-frame nominal f0 for each voice (NaN = voice inactive).

    Extends tracks with `pad` frames and interpolates over short internal
    gaps so that masks do not drop to zero between closely-spaced tracks.
    """
    f0 = np.full((n_voices, n_frames), np.nan, dtype=np.float32)
    for tr, v in zip(tracks, labels):
        # Build dense arrays over [start_frame, end_frame]
        frame_set = {f: i for i, f in enumerate(tr.frames)}
        dense_frames = list(range(tr.start_frame, tr.end_frame + 1))
        dense_f0 = []
        last_known = None
        gap_start = None
        for f in dense_frames:
            if f in frame_set:
                hz = tr.f0_hz[frame_set[f]]
                dense_f0.append(hz)
                last_known = hz
                if gap_start is not None:
                    # We just ended a gap: interpolate it
                    gap_end = len(dense_f0) - 1
                    gap_len = gap_end - gap_start
                    if gap_len <= max_interp_gap and gap_len > 0:
                        # Need start value before gap; if none, use last_known
                        start_hz = dense_f0[gap_start - 1] if gap_start > 0 else last_known
                        for k in range(gap_start, gap_end):
                            alpha = (k - gap_start + 1) / (gap_len + 1)
                            dense_f0[k] = start_hz + alpha * (last_known - start_hz)
                    gap_start = None
            else:
                dense_f0.append(np.nan)
                if gap_start is None:
                    gap_start = len(dense_f0) - 1

        # Pad with held f0 (not NaN) to avoid mask dropouts at note edges
        padded_start = max(tr.start_frame - pad, 0)
        padded_end = min(tr.end_frame + pad + 1, n_frames)
        for f in range(padded_start, padded_end):
            if f < tr.start_frame:
                src = tr.start_frame
                src_idx = src - tr.start_frame
                val = dense_f0[src_idx] if src_idx < len(dense_f0) else tr.f0_hz[0]
            elif f > tr.end_frame:
                src = tr.end_frame
                src_idx = src - tr.start_frame
                val = dense_f0[src_idx] if src_idx < len(dense_f0) else tr.f0_hz[-1]
            else:
                src_idx = f - tr.start_frame
                val = dense_f0[src_idx] if src_idx < len(dense_f0) else np.nan
            if not np.isnan(val):
                f0[v, f] = val
    return f0

# ---------- 4. Audio splitting ----------
def refine_f0(mag, f0, sr, n_fft, H=10, cents=50):
    """Snap the nominal f0 to the strongest harmonic comb within +-cents."""
    bin_hz = sr / n_fft
    ratios = 2 ** (np.arange(-cents, cents + 1, 2) / 1200)
    k = np.arange(1, H + 1)
    out = f0.copy()
    for v in range(f0.shape[0]):
        for t in np.where(~np.isnan(f0[v]))[0]:
            cand = f0[v, t] * ratios
            idx = np.clip(np.rint(cand[:, None] * k / bin_hz).astype(int), 0, mag.shape[0] - 1)
            out[v, t] = cand[np.argmax(mag[idx, t].sum(axis=1))]
    return out


def harmonic_masks(f0, mag, sr, n_fft, H=25, hw0=45.0, hwr=0.025,
                   smooth_frames: int = 5, min_mask: float = 0.10,
                   shared_min: float = 0.20):
    """Soft harmonic masks with shared-harmonic protection.

    Wider Gaussians capture more energy around each harmonic. When two active
    voices both have a harmonic near the same bin, neither is forced below
    `shared_min`, preventing one voice from disappearing in harmonies.
    """
    V, T = f0.shape
    n_bins = n_fft // 2 + 1
    freqs = np.arange(n_bins) * sr / n_fft
    bin_hz = sr / n_fft
    k = np.arange(1, H + 1)
    m = np.zeros((V, n_bins, T), np.float32)
    for v in range(V):
        for t in np.where(~np.isnan(f0[v]))[0]:
            c = f0[v, t] * k
            sigma = (hw0 + hwr * c) / 2
            g = np.exp(-0.5 * ((freqs[:, None] - c[None, :]) / sigma) ** 2)
            m[v, :, t] = g.max(axis=1)

    # Weight by observed magnitude so strong harmonics get priority
    mag_t = mag[None, :, :].astype(np.float32)
    E = m * (1.0 + 0.5 * (mag_t / (mag_t.max() + 1e-8)))

    act = (~np.isnan(f0)).astype(np.float32)
    A = np.maximum(act.sum(axis=0), 1)
    share = act / A[None, :]

    # Base soft-max mask plus shared residual
    beta = 0.10
    den = E.sum(axis=0, keepdims=True) + beta * (A > 0)[None, None, :]
    masks = (E + beta * share[:, None, :]) / np.maximum(den, 1e-8)

    # Shared-harmonic protection: when two voices are active and both have a
    # harmonic within 1.5 bins, ensure neither mask drops below shared_min.
    if V >= 2:
        for t in np.where(A > 1)[0]:
            # distance from each bin to nearest harmonic for each voice
            dist = np.full((V, n_bins), np.inf, dtype=np.float32)
            for v in range(V):
                if np.isnan(f0[v, t]):
                    continue
                harm_bins = np.clip(np.rint(f0[v, t] * k / bin_hz).astype(int), 0, n_bins - 1)
                for hb in harm_bins:
                    dist[v, :] = np.minimum(dist[v, :], np.abs(np.arange(n_bins) - hb))
            shared = np.all(dist < 1.5, axis=0)
            if shared.any():
                for v in range(V):
                    low = shared & (masks[v, :, t] < shared_min)
                    if low.any():
                        # Boost the weak voice and reduce the dominant one
                        other = 1 - v
                        transfer = np.minimum(masks[other, low, t] - shared_min, shared_min - masks[v, low, t])
                        transfer = np.maximum(transfer, 0)
                        masks[v, low, t] += transfer
                        masks[other, low, t] -= transfer

    # Temporal smoothing reduces abrupt switches between voices
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        for v in range(V):
            for b in range(n_bins):
                masks[v, b, :] = np.convolve(masks[v, b, :], kernel, mode='same')

    # Floor for active voices
    floor = min_mask * act[:, None, :]
    masks = np.maximum(masks, floor)
    masks = masks / np.maximum(masks.sum(axis=0, keepdims=True), 1e-8)
    return masks


def split_harmonic(vocals: Path, notes, labels, n_voices, out_stem: Path,
                   n_fft=4096, hop=1024):
    y, sr = librosa.load(vocals, sr=None, mono=False)
    y = y[None] if y.ndim == 1 else y
    S = [librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in y]
    mag = np.mean([np.abs(s) for s in S], axis=0)

    # Build f0 from notes (legacy support)
    f0 = np.full((n_voices, mag.shape[1]), np.nan, dtype=np.float32)
    for (s, e, p, _), v in zip(notes, labels):
        a = int(s * sr / hop) - 2
        b = int(np.ceil(e * sr / hop)) + 2
        f0[v, max(a, 0):min(b, mag.shape[1])] = librosa.midi_to_hz(p)

    f0 = refine_f0(mag, f0, sr, n_fft)
    masks = harmonic_masks(f0, mag, sr, n_fft)
    for v in range(n_voices):
        chans = [librosa.istft(s * masks[v], hop_length=hop, length=y.shape[1]) for s in S]
        path = out_stem.parent / f"{out_stem.name}_voice{v+1}.wav"
        sf.write(path, np.stack(chans, axis=1), sr)
        print("wrote", path)


def split_harmonic_from_tracks(vocals: Path, tracks: List[PitchTrack],
                               labels: List[int], n_voices: int,
                               out_stem: Path, segments=None,
                               n_fft=4096, hop=1024,
                               sr_target: Optional[int] = None):
    """Split audio using continuous pitch tracks."""
    y, sr = librosa.load(vocals, sr=sr_target, mono=False)
    y = y[None] if y.ndim == 1 else y
    S = [librosa.stft(ch, n_fft=n_fft, hop_length=hop) for ch in y]
    mag = np.mean([np.abs(s) for s in S], axis=0)

    f0 = f0_tracks_from_contours(tracks, labels, n_voices, mag.shape[1], sr, hop)
    f0 = fill_f0_by_diarization(f0, segments, n_voices, sr, hop)
    f0 = refine_f0(mag, f0, sr, n_fft)
    masks = harmonic_masks(f0, mag, sr, n_fft)
    for v in range(n_voices):
        chans = [librosa.istft(s * masks[v], hop_length=hop, length=y.shape[1]) for s in S]
        path = out_stem.parent / f"{out_stem.name}_voice{v+1}.wav"
        sf.write(path, np.stack(chans, axis=1), sr)
        print("wrote", path)

# ---------- 5. Diagnostics ----------
class Diagnostics:
    def __init__(self, sr: int, hop: int, times: np.ndarray, freqs: np.ndarray,
                 mag: np.ndarray):
        self.sr = sr
        self.hop = hop
        self.times = times
        self.freqs = freqs
        self.mag = mag
        self.tracks = []
        self.labels = []
        self.segments = None
        self.embeddings_timeline = None
        self.n_voices = 2

    def set_tracks(self, tracks: List[PitchTrack], labels: List[int]):
        self.tracks = tracks
        self.labels = labels

    def set_segments(self, segments, embeddings_timeline):
        self.segments = segments
        self.embeddings_timeline = embeddings_timeline

    def save(self, diag_dir: Path):
        diag_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(diag_dir / "tracks.json")
        self._plot_overview(diag_dir / "overview.png")
        self._plot_masks(diag_dir / "masks.png")
        self._plot_metrics(diag_dir / "metrics.png")
        print(f"Saved diagnostics to {diag_dir}")

    def _save_json(self, path: Path):
        data = {
            "sr": self.sr,
            "hop": self.hop,
            "n_frames": len(self.times),
            "n_voices": self.n_voices,
            "tracks": [
                {
                    "track_id": tr.track_id,
                    "voice": int(self.labels[i]) if i < len(self.labels) else -1,
                    "start_frame": int(tr.start_frame),
                    "end_frame": int(tr.end_frame),
                    "mean_midi": float(tr.mean_midi()),
                    "mean_confidence": float(tr.mean_confidence()),
                }
                for i, tr in enumerate(self.tracks)
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _plot_overview(self, path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1, 1]})

        # 1. Spectrogram + tracks
        ax = axes[0]
        spec_db = librosa.amplitude_to_db(self.mag, ref=np.max)
        im = ax.imshow(spec_db, aspect="auto", origin="lower",
                       extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]],
                       cmap="magma", vmin=-80, vmax=0)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        colors = ["cyan", "lime", "yellow", "magenta"]
        for tr, lab in zip(self.tracks, self.labels):
            c = colors[lab % len(colors)]
            ax.plot([self.times[f] for f in tr.frames], tr.f0_hz, color=c, lw=1.5, alpha=0.8)
        ax.set_ylabel("Hz")
        ax.set_title("Spectrogram + tracked F0 contours (colored by voice)")
        ax.set_ylim(self.freqs[0], min(8000, self.freqs[-1]))

        # 2. Speaker segments
        ax = axes[1]
        if self.segments:
            for seg_s, seg_e, spk in self.segments:
                c = colors[spk % len(colors)]
                ax.barh(spk, seg_e - seg_s, left=seg_s, height=0.6, color=c, alpha=0.6)
        ax.set_ylabel("speaker")
        ax.set_title("Speaker diarization segments")
        ax.set_yticks(range(self.n_voices))

        # 3. Confidence per frame per voice
        ax = axes[2]
        confidences = [np.zeros(len(self.times)) for _ in range(self.n_voices)]
        for tr, lab in zip(self.tracks, self.labels):
            for fi, conf in zip(tr.frames, tr.confidence):
                if 0 <= fi < len(self.times):
                    confidences[lab][fi] = max(confidences[lab][fi], conf)
        for v in range(self.n_voices):
            ax.plot(self.times, confidences[v], color=colors[v], lw=1, alpha=0.8)
        ax.set_ylabel("confidence")
        ax.set_title("Track confidence per frame")
        ax.set_ylim(0, 1)

        # 4. Voice activity
        ax = axes[3]
        activity = [np.zeros(len(self.times)) for _ in range(self.n_voices)]
        for tr, lab in zip(self.tracks, self.labels):
            for fi in tr.frames:
                if 0 <= fi < len(self.times):
                    activity[lab][fi] = 1
        for v in range(self.n_voices):
            ax.fill_between(self.times, activity[v] + v * 1.2, v * 1.2,
                            color=colors[v], alpha=0.6, step="mid")
        ax.set_ylabel("voice")
        ax.set_xlabel("time (s)")
        ax.set_title("Detected voice activity")
        ax.set_yticks([v * 1.2 + 0.5 for v in range(self.n_voices)])
        ax.set_yticklabels([f"V{v+1}" for v in range(self.n_voices)])

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_masks(self, path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build f0 and masks from tracks
        f0 = f0_tracks_from_contours(self.tracks, self.labels, self.n_voices,
                                     len(self.times), self.sr, self.hop)
        f0 = fill_f0_by_diarization(f0, self.segments, self.n_voices, self.sr, self.hop)
        f0 = refine_f0(self.mag, f0, self.sr, (len(self.freqs) - 1) * 2)
        masks = harmonic_masks(f0, self.mag, self.sr, (len(self.freqs) - 1) * 2)

        fig, axes = plt.subplots(self.n_voices + 1, 1, figsize=(16, 10), sharex=True)
        for v in range(self.n_voices):
            ax = axes[v]
            im = ax.imshow(masks[v], aspect="auto", origin="lower",
                           extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]],
                           cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            ax.set_ylabel("Hz")
            ax.set_title(f"Voice {v+1} mask")
            ax.set_ylim(self.freqs[0], min(8000, self.freqs[-1]))

        ax = axes[-1]
        overlap = np.sum(masks > 0.3, axis=0)
        im = ax.imshow(overlap, aspect="auto", origin="lower",
                       extent=[self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]],
                       cmap="hot", vmin=0, vmax=self.n_voices)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        ax.set_ylabel("Hz")
        ax.set_xlabel("time (s)")
        ax.set_title("Number of voices claiming each bin (>0.3)")
        ax.set_ylim(self.freqs[0], min(8000, self.freqs[-1]))

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    def _plot_metrics(self, path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_frames = len(self.times)
        dt = self.times[1] - self.times[0] if n_frames > 1 else self.hop / self.sr

        # Compute gaps per voice
        gaps = []
        for v in range(self.n_voices):
            active = np.zeros(n_frames)
            for tr, lab in zip(self.tracks, self.labels):
                if lab != v:
                    continue
                for fi in tr.frames:
                    if 0 <= fi < n_frames:
                        active[fi] = 1
            # Find contiguous zero regions
            gap_lengths = []
            cur = 0
            for a in active:
                if a == 0:
                    cur += 1
                else:
                    if cur > 0:
                        gap_lengths.append(cur * dt)
                        cur = 0
            if cur > 0:
                gap_lengths.append(cur * dt)
            gaps.append(gap_lengths)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.hist([g for gl in gaps for g in gl], bins=50, color="steelblue", edgecolor="black")
        ax.set_xlabel("gap duration (s)")
        ax.set_ylabel("count")
        ax.set_title("Distribution of voice dropout gaps")

        ax = axes[0, 1]
        for v, gl in enumerate(gaps):
            ax.ecdf(gl, label=f"Voice {v+1}")
        ax.set_xlabel("gap duration (s)")
        ax.set_ylabel("CDF")
        ax.set_title("Cumulative dropout gaps")
        ax.legend()

        ax = axes[1, 0]
        n_overlapping = 0
        for i in range(n_frames):
            active = sum(1 for tr, lab in zip(self.tracks, self.labels)
                         if i in tr.frames)
            if active >= 2:
                n_overlapping += 1
        labels = ["solo", "overlap"]
        values = [n_frames - n_overlapping, n_overlapping]
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title("Frames with 1 vs 2+ detected F0")

        ax = axes[1, 1]
        # Track length distribution
        lengths = [(tr.end_frame - tr.start_frame + 1) * dt for tr in self.tracks]
        ax.hist(lengths, bins=50, color="coral", edgecolor="black")
        ax.set_xlabel("track duration (s)")
        ax.set_ylabel("count")
        ax.set_title("Distribution of pitch-track lengths")

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)


# ---------- 6. MIDI / CSV ----------
def write_outputs(tracks: List[PitchTrack], labels: List[int], n_voices: int,
                  stem: Path, dt: float):
    pm = pretty_midi.PrettyMIDI()
    insts = [pretty_midi.Instrument(program=52, name=f"voice_{v+1}") for v in range(n_voices)]
    for tr, v in zip(tracks, labels):
        s, e, midi, amp = tr.to_note(dt)
        insts[v].notes.append(pretty_midi.Note(
            velocity=int(np.clip(amp * 127, 30, 127)), pitch=midi, start=s, end=e))
    pm.instruments.extend(insts)
    pm.write(str(stem.with_suffix(".mid")))
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["voice", "start_s", "end_s", "midi", "note"])
        rows = []
        for tr, v in zip(tracks, labels):
            s, e, midi, _ = tr.to_note(dt)
            rows.append((s, e, midi, v))
        for s, e, p, v in sorted(rows, key=lambda x: x[0]):
            w.writerow([v + 1, f"{s:.3f}", f"{e:.3f}", p, pretty_midi.note_number_to_name(p)])


# ---------- 7. Legacy compatibility ----------
def notes_poly(vocals: Path, min_dur, fmin, fmax):
    """Deprecated: kept for backwards compatibility."""
    cands, _, times, _ = detect_pitch_candidates(vocals, fmin, fmax)
    tracks = track_pitches(cands)
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    notes = []
    for tr in tracks:
        s, e, midi, amp = tr.to_note(dt)
        notes.append((s, e, midi, amp))
    return notes


def f0_tracks(notes, labels, n_voices, n_frames, sr, hop, pad=2):
    """Legacy compatibility."""
    f0 = np.full((n_voices, n_frames), np.nan)
    for (s, e, p, _), v in zip(notes, labels):
        a = int(s * sr / hop) - pad
        b = int(np.ceil(e * sr / hop)) + pad
        f0[v, max(a, 0):min(b, n_frames)] = librosa.midi_to_hz(p)
    return f0


def assign_voices(notes, segments=None, n_voices=2):
    """Legacy compatibility for old note-based assignment."""
    if n_voices == 1 or not notes:
        return [0] * len(notes)
    # Convert legacy notes to tracks
    dt = 0.01
    tracks = []
    for i, (s, e, p, a) in enumerate(notes):
        start_frame = int(s / dt)
        end_frame = int(e / dt)
        tr = PitchTrack(track_id=i, start_frame=start_frame, end_frame=end_frame)
        tr.frames = list(range(start_frame, end_frame + 1))
        tr.f0_hz = np.full(len(tr.frames), librosa.midi_to_hz(p))
        tr.magnitude = np.full(len(tr.frames), a)
        tr.confidence = np.full(len(tr.frames), a)
        tracks.append(tr)
    labels = assign_track_identities(tracks, segments, None, n_voices, None, 22050, 256)
    return labels


# ---------- 8. Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song", type=Path)
    ap.add_argument("--split", choices=["harmonic", "stereo"], default="harmonic")
    ap.add_argument("--mode", choices=["poly", "pyin"], default="poly")
    ap.add_argument("--voices", type=int, default=2)
    ap.add_argument("--model", default="htdemucs_ft")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--min-dur", type=float, default=0.05, help="seconds")
    ap.add_argument("--fmin", default="C2")
    ap.add_argument("--fmax", default="C6")
    ap.add_argument("--diarizer", choices=["pyannote", "speechbrain", "none"],
                    default="pyannote", help="speaker diarization backend")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for pyannote (else uses $HF_TOKEN)")
    ap.add_argument("--max-pitches", type=int, default=4,
                    help="max simultaneous pitches per frame (multi-pitch only)")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="skip diagnostic plots")
    a = ap.parse_args()

    a.out.mkdir(exist_ok=True)
    stem = a.out / a.song.stem
    vocals = separate(a.song, a.out, a.model)

    if a.split == "stereo":
        split_stereo(vocals, stem)
        return

    if a.mode == "pyin" or a.voices < 2:
        sys.exit("Harmonic split needs --mode poly and --voices >= 2.")

    print(f"[1/5] Multi-pitch candidate detection...")
    candidates, mag, times, freqs = detect_pitch_candidates(
        vocals, a.fmin, a.fmax, max_pitches=a.max_pitches)
    print(f"      {len(candidates)} candidates in {len(set(c.frame_idx for c in candidates))} frames")

    print(f"[2/5] Multi-pitch tracking...")
    tracks = track_pitches(candidates, max_voices=a.voices,
                           min_dur=a.min_dur)
    print(f"      {len(tracks)} tracks")

    print(f"[3/5] Speaker diarization ({a.diarizer})...")
    segments, embeddings_timeline = diarize_speakers(
        vocals, n_speakers=a.voices, backend=a.diarizer, hf_token=a.hf_token)
    if segments is None:
        print("      no diarization, using pitch-continuity assignment")

    print(f"[4/5] Voice assignment...")
    labels = assign_track_identities(tracks, segments, embeddings_timeline,
                                     a.voices, vocals, 22050, 256)
    for v in range(a.voices):
        n_v = sum(1 for l in labels if l == v)
        print(f"      voice {v+1}: {n_v} tracks")

    print(f"[5/5] Writing outputs and splitting audio...")
    dt = times[1] - times[0] if len(times) > 1 else 256 / 22050
    write_outputs(tracks, labels, a.voices, stem, dt)
    split_harmonic_from_tracks(vocals, tracks, labels, a.voices, stem,
                               segments=segments)

    if not a.no_diagnostics:
        diag = Diagnostics(22050, 256, times, freqs, mag)
        diag.n_voices = a.voices
        diag.set_tracks(tracks, labels)
        diag.set_segments(segments, embeddings_timeline)
        diag_dir = a.out / f"{a.song.stem}_diagnostics"
        diag.save(diag_dir)

    print(f"Done: {len(tracks)} tracks -> {stem}.mid / .csv / _voice*.wav")


if __name__ == "__main__":
    main()
