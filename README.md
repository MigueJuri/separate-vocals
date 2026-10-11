# SeparateVocals

Multi-voice vocal separation, polyphonic pitch tracking and per-voice audio
splitting for music. Given a stereo song, the pipeline isolates the vocal
stem with [Demucs](https://github.com/facebookresearch/demucs), detects
simultaneous F0 candidates with librosa, tracks each voice with a
Hungarian-assignment tracker, anchors tracks to speaker identity through
diarization, and writes per-voice `.wav` stems, a `.mid` score and a `.csv`
note listing.

The project demonstrates an end-to-end audio-MIR pipeline that can be
followed from raw audio to structured, exportable musical data.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [What it does](#what-it-does)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [How the pipeline works](#how-the-pipeline-works)
- [Outputs](#outputs)
- [Diarization backends](#diarization-backends)
- [Testing](#testing)
- [Performance notes](#performance-notes)
- [Tech stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why this exists

Source separation models such as Demucs return a *single* vocal stem even
when two singers are clearly present. The remaining problem is to turn that
mono vocal into per-voice stems with correct identity continuity across
time, even when voices overlap in pitch and in time. This project tackles
that problem with a fully local, open-source Python pipeline:

1. Separate the vocal stem from a song.
2. Detect multiple simultaneous pitches per frame with hysteresis and
   harmonic-strength scoring.
3. Track the candidate pitches across frames with a Hungarian-assignment
   tracker that handles crossings and short dropouts.
4. Anchor each track to a speaker using either
   [pyannote.audio](https://github.com/pyannote/pyannote-audio) or
   [speechbrain](https://github.com/speechbrain/speechbrain) embeddings, and
   resolve collisions when two tracks claim the same voice.
5. Build per-voice harmonic masks, refine the F0, and reconstruct a
   time-aligned stem for each voice.
6. Export the result as `.wav`, `.mid` and `.csv`, plus diagnostic plots.

It is a self-contained reference implementation suitable for prototyping,
research, and education. No proprietary models or services are required.

## What it does

| Stage | What happens |
| --- | --- |
| 1. Separation | Demucs (`htdemucs_ft` by default) writes `vocals.wav` and `no_vocals.wav`. |
| 2. Multi-pitch | `piptrack` + hysteresis + harmonic-strength scoring produces up to 4 pitch candidates per frame. |
| 3. Tracking | Hungarian assignment per frame, with gap interpolation and short-track merging. |
| 4. Diarization | Speaker turns are detected by pyannote (preferred) or speechbrain + spectral clustering. |
| 5. Identity | Tracks are assigned to voices via speaker overlap, embeddings, and collision resolution. |
| 6. Masking | F0 is refined, per-voice harmonic masks are built with shared-harmonic protection, and voices are reconstructed with inverse STFT. |
| 7. Export | `voice1.wav`, `voice2.wav`, `song.mid`, `song.csv`, plus `overview.png`, `masks.png`, `metrics.png` and `tracks.json`. |

## Repository layout

```
.
├── src/separatevocals/      # Importable package (separation, init, entry point)
├── tests/                   # Unit tests for the pipeline building blocks
├── docs/                    # Architecture notes and design decisions
├── examples/                # Example commands and configuration snippets
├── .github/workflows/       # CI: lint, test, syntax check
├── vocal_notes.py           # Original CLI script (kept for reference)
├── pyproject.toml           # Project metadata, dependencies, console entry point
├── requirements.txt         # Plain-pip dependency list
├── LICENSE                  # MIT
└── README.md                # You are here
```

## Installation

Tested on Python 3.11 with CUDA 12.x. Demucs will run on CPU but is
significantly faster with a recent NVIDIA GPU.

```bash
# 1. Clone the repository
git clone https://github.com/MigueJuri/separate-vocals.git
cd separate-vocals

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package with the diarization extras
pip install -e ".[diarization,dev]"

# 4. (pyannote only) accept licenses and set a token
#    https://huggingface.co/pyannote/segmentation-3.0
#    https://huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=hf_xxx
```

The first Demucs run will download the `htdemucs_ft` weights
(~80 MB) into `~/.cache/torch/hub/`.

## Quick start

```bash
# Multi-pitch mode with pyannote diarization (recommended)
separate-vocals path/to/song.mp3

# Token-free fallback: speechbrain + spectral clustering
separate-vocals path/to/song.mp3 --diarizer speechbrain

# Legacy pitch-continuity (no diarization at all)
separate-vocals path/to/song.mp3 --diarizer none

# Hard-panned stereo split (L/R as two voices)
separate-vocals path/to/song.mp3 --split stereo

# Solo voice: single-pitch pYIN
separate-vocals path/to/song.mp3 --mode pyin
```

Outputs land in `./out/<song_stem>/...` by default. See
[Outputs](#outputs) for the full layout.

## CLI reference

```
separate-vocals SONG [options]

positional:
  SONG                     Path to an input audio file (mp3, wav, flac, ...).

options:
  --split {harmonic,stereo}
                          How to split the vocal stem. Default: harmonic.
  --mode {poly,pyin}      Pitch detection mode. Default: poly.
  --voices N              Number of voices to track. Default: 2.
  --model NAME            Demucs model. Default: htdemucs_ft.
  --out DIR               Output directory. Default: ./out.
  --min-dur SECONDS       Drop tracks shorter than this. Default: 0.05.
  --fmin NOTE             Lower bound for pitch detection. Default: C2.
  --fmax NOTE             Upper bound for pitch detection. Default: C6.
  --diarizer {pyannote,speechbrain,none}
                          Speaker diarization backend. Default: pyannote.
  --hf-token TOKEN        HuggingFace token for pyannote (else uses $HF_TOKEN).
  --max-pitches N         Max simultaneous pitches per frame. Default: 4.
  --no-diagnostics        Skip diagnostic plots.
```

## How the pipeline works

1. **Separation.** Demucs is invoked via `python -m demucs --two-stems=vocals`
   so the same model weights are reused if a previous run already produced a
   `vocals.wav`. The library wrapper around this call lives in
   `src/separatevocals/separation.py`.

2. **Multi-pitch candidates.** `librosa.piptrack` gives a raw pitch estimate
   for every STFT bin. We then keep local maxima, score each peak by its
   magnitude relative to the frame maximum (with hysteresis) and by its
   harmonic strength, i.e. how much of the frame's energy lies on the first
   `H` harmonics. Up to `max_pitches` candidates are kept per frame.

3. **Tracking.** For each frame we build a cost matrix between the active
   tracks and the current candidates. Cost is a sum of cents distance and a
   confidence penalty. We solve the assignment with `scipy.optimize.
   linear_sum_assignment` (Hungarian), so a single global decision resolves
   crossings. Short gaps are forward-filled, short tracks are dropped, and
   nearby fragments are merged.

4. **Diarization + identity assignment.** The vocal stem is fed to a
   diarization backend that returns `(start, end, speaker_id)` turns and, if
   possible, a time series of speaker embeddings. Tracks are first assigned
   to a voice if a single speaker dominates its overlap (>70%). Remaining
   tracks are assigned by cosine similarity to voice prototypes built from
   strongly-dominating tracks, with a final pitch-continuity fallback. A
   collision pass swaps voices when two tracks overlap in time. Voices are
   remapped so that voice 1 is the higher mean pitch.

5. **F0 and masks.** F0 is interpolated inside short track gaps, padded by
   a few frames at each edge, and snapped to the strongest harmonic comb.
   Per-voice masks are built with wide harmonic Gaussians, weighted by the
   observed magnitude, and protected from collapse in regions where two
   voices share a harmonic.

6. **Reconstruction.** The mask is applied to each channel of the vocal
   stem's STFT and the signal is reconstructed with `librosa.istft`.

A deeper write-up of the design trade-offs lives in
[`docs/architecture.md`](docs/architecture.md).

## Outputs

After a successful run with `song = path/to/bendita_tu_luz.mp3`:

```
out/
  htdemucs_ft/bendita_tu_luz/
    vocals.wav
    no_vocals.wav
  bendita_tu_luz.mid
  bendita_tu_luz.csv
  bendita_tu_luz_voice1.wav
  bendita_tu_luz_voice2.wav
  bendita_tu_luz_diagnostics/
    tracks.json
    overview.png
    masks.png
    metrics.png
```

- `voice{1,2}.wav` — per-voice reconstruction, same length and sample rate
  as the input.
- `*.mid` — `pretty_midi` score with one program-52 instrument per voice.
- `*.csv` — `voice,start_s,end_s,midi,note` per row.
- `tracks.json` — track id, voice label, frame range, mean midi and mean
  confidence.
- `overview.png` — spectrogram + tracked F0 + speaker segments + activity.
- `masks.png` — per-voice harmonic mask + overlap heatmap.
- `metrics.png` — dropout gap histograms and solo/overlap pie.

## Diarization backends

| Backend | Token | Quality | Notes |
| --- | --- | --- | --- |
| `pyannote` | required | best | Needs `HF_TOKEN` and accepted pyannote licenses. |
| `speechbrain` | not required | good | ECAPA-TDNN embeddings + spectral clustering; self-contained. |
| `none` | — | weakest | Falls back to legacy pitch-continuity assignment. |

If `pyannote.audio` is not installed, the pipeline silently falls back to
`speechbrain`.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The tests cover pure-python / numpy pieces (Hungarian assignment, hysteresis
thresholds, F0 padding and harmonic mask building) so they can run quickly
in CI without GPU or downloaded model weights.

## Performance notes

- Demucs separation on `htdemucs_ft` is the dominant cost. On a modern CPU
  expect roughly real-time for a 3-minute song; on a 30-series GPU expect
  10–20x real-time.
- Multi-pitch detection scales with `n_frames * n_voices` and is well under
  1s for typical songs.
- pyannote diarization is the second dominant cost (tens of seconds on CPU,
  a few seconds on GPU).

## Tech stack

- Python 3.11, NumPy, SciPy, librosa, soundfile
- Demucs 4.x, PyTorch 2.x
- pyannote.audio 3.x and / or speechbrain 1.x
- pretty_midi for MIDI export
- matplotlib for diagnostics
- pytest, ruff and mypy for development

## Roadmap

- [ ] Replace Hungarian assignment with a probabilistic tracker.
- [ ] Train a small learned pitch-presence classifier.
- [ ] Add a stereo-panned variant of the harmonic mask.
- [ ] Add a web demo (Gradio) for non-technical users.
- [ ] Add CI matrix across Linux / macOS and Python 3.10–3.12.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI Research.
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) by Hervé Bredin.
- [speechbrain](https://github.com/speechbrain/speechbrain).
- [librosa](https://librosa.org/) and [pretty_midi](https://github.com/craffel/pretty-midi).
