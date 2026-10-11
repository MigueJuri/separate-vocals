# Architecture

This document explains the design decisions behind the
`separate-vocals` pipeline. It is intended for reviewers who want to
understand the trade-offs without reading the code top to bottom.

## Goals

- **Faithful voice identity.** A note sung by voice 1 must remain in
  `voice1.wav` for its entire duration, even if voice 2 sings the same
  pitch at the same time.
- **Polyphony.** The tracker must handle two simultaneous pitches
  crossing in pitch space (voice 1 goes up while voice 2 goes down).
- **Robustness.** Short detection dropouts must not fragment tracks.
- **Reproducibility.** No cloud services; everything runs locally.

## Pipeline

```
song.mp3
   |
   |  (Demucs htdemucs_ft, two-stems=vocals)
   v
vocals.wav
   |
   |  STFT + librosa.piptrack + hysteresis + harmonic-strength
   v
List[PitchCandidate]    # per-frame multi-pitch
   |
   |  Hungarian assignment per frame + gap interpolation + merging
   v
List[PitchTrack]
   |
   |  Speaker diarization (pyannote or speechbrain)
   v
segments, embeddings
   |
   |  Speaker-overlap + embedding similarity + collision resolution
   v
labels: List[int]       # voice index per track
   |
   |  F0 padding + harmonic-combing + harmonic masking
   v
masks: (n_voices, n_bins, n_frames)
   |
   |  istft per voice
   v
voice1.wav, voice2.wav
```

## Why Demucs?

Demucs's `htdemucs_ft` is a strong, open-source vocal separator. The
interface is intentionally simple: we shell out to `python -m demucs`
so that the wrapper does not need to track Demucs internals across
versions. If a `vocals.wav` already exists, we skip the call and reuse
it; this makes the pipeline cheap to re-run with different pitch
parameters.

## Why per-frame Hungarian assignment?

A greedy "nearest-pitch" tracker can swap identities when two voices
cross in pitch. A global optimum per frame, via
`linear_sum_assignment`, prevents that swap. Cost is a sum of cents
distance and a confidence penalty:

```
cost[i, j] = cents(last_f0_i, f0_j) + 50 * (1 - confidence_j)
```

A large value (1e9) is used as the "no link" cost so that the
Hungarian solver can still pick a high-cost match when it improves the
overall assignment.

## Why diarization + embeddings?

Speaker overlap alone is a noisy signal: two voices often share a
microphone region. We combine three signals:

1. **Overlap with diarized turns.** A track that overlaps 95% with
   speaker 0 and 5% with speaker 1 is almost certainly speaker 0.
2. **Cosine similarity to voice prototypes.** Built from the strongly
   dominating tracks above, the prototypes act as a centroid for each
   voice.
3. **Pitch continuity.** When all else fails we assign the remaining
   tracks by nearest previous pitch, then resolve collisions at
   overlapping frames by swapping the weaker track.

A final pass re-orders voices so voice 1 has the higher mean pitch.
That makes the output more interpretable for downstream tooling.

## Why soft harmonic masks?

A hard mask (keep bins close to voice 1's harmonic) is brittle:
octave errors and partial overlaps sound like "robot voice" once
reconstructed. A wide Gaussian around each harmonic is more forgiving
and, weighted by the observed magnitude, naturally handles
shared-harmonic regions. The `shared_min` floor prevents a stronger
voice from erasing a weaker one in unison passages.

## Why a single-file CLI plus a thin package?

The original `vocal_notes.py` is large enough that splitting it
across many modules would be a refactor, not a feature. Instead we:

- keep `vocal_notes.py` as a self-contained reference implementation,
- expose the same logic through `src/separatevocals/` with a
  `console_scripts` entry point named `separate-vocals`,
- let the package re-export `vocal_notes` so existing call sites
  continue to work.

This keeps the diff small while still giving the project a proper
installable surface.
